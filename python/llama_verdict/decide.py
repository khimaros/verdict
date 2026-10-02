"""the decision api: one prefill per state, one short suffix per question."""

import dataclasses
import functools
import hashlib
import json
import random
import time

from . import backend as backend_mod
from . import derive, extract, joint_prompt, prompt, spec, types
from . import head as heads

# how many tokenized prompt pieces a decider remembers
TOKEN_CACHE = 4096


class Decider:
    """binds a backend to a formatter and answers typed questions about a state."""

    def __init__(self, backend, formatter, tournament=False, pretokenize=False,
                 order_averaging=1, prior_correction=False, wide_alphabet=False,
                 cache_dir=derive.CACHE_DIR, head=None):
        formatter.check_usable()
        self.backend = backend
        self.formatter = formatter
        # a layout that names a decision head is read through it: the hidden
        # state at the scored position, and the head's rows instead of labels
        self.head = head or heads.for_layout(formatter.layout)
        # a joint prompt is tokenized piece by piece and most pieces repeat
        self._tokens_of = functools.lru_cache(maxsize=TOKEN_CACHE)(
            lambda text: tuple(backend.tokenize(text)))
        self._layout = spec.layout(formatter.layout)
        # send prompts as text by default: it is the same token sequence
        # wherever the spec's boundary assertion holds, and it costs no
        # /tokenize round trip
        self.pretokenize = pretokenize
        self._prefix_cache = (None, None)
        # two ways to answer a list past the label ceiling, both opt-in.
        # the tournament returns an approximation and must never appear without
        # the caller having asked for it; the wide alphabet is exact, and asks
        # because resolving it costs a whole-vocabulary read on first use.
        self.tournament = tournament
        self.wide_alphabet = wide_alphabet
        self.cache_dir = cache_dir
        self._wide = dict(formatter.wide_label_ids)
        self._wide_checked = None
        # debiasing. both cost extra passes, so both are opt-in.
        self.order_averaging = max(1, order_averaging)
        self.prior_correction = prior_correction
        self._prior_cache = {}
        # seeded from the formatter, which pinned them at derivation time
        self._label_ids = dict(formatter.label_ids)

    @classmethod
    def from_server(cls, base_url, model, formatter_path, **kw):
        return cls(backend_mod.HttpBackend(base_url, model),
                   types.Formatter.load(formatter_path), **kw)

    def _score_options(self, prefix, question, options):
        """one scoring pass over one option list.

        the prompt goes as text rather than as separately tokenised halves.
        the spec requires `tokenize(prefix) + tokenize(suffix)` to equal
        `tokenize(prefix + suffix)` and treats a mismatch as a hard error, so
        where that holds the two are the same token sequence and the text form
        simply costs no round trip. text is also what makes a request legible
        in a proxy log, which is where these actually get debugged.

        `pretokenize=True` is a VERIFICATION SWITCH, not a serving option: it
        restores the explicit split so the boundary assertion can be checked
        per request. nothing in the serving path sets it, and
        `python/tests/test_wire_format.py` holds that.
        """
        sub = dataclasses.replace(question, options=tuple(options))
        alphabet = self._alphabet(len(options))
        suffix = prompt.render_suffix(self.formatter, sub, alphabet)
        if self.head:
            scored = self.backend.hidden(prefix + suffix)
            probs = self.head.probs(scored.pop("hidden"), len(options))
            return ({o.id: p for o, p in zip(options, probs, strict=True)}, scored, suffix)
        labels = prompt.label_map(sub, alphabet, self.formatter.layout)
        ids = self.label_ids(list(labels.values()))

        # where the shared state ends, so the server checkpoints it and the next
        # question over the same state resumes instead of refilling it
        anchor = self.formatter.checkpoint_anchor
        delimiters = [{"role": "user", "delimiter": anchor}] if anchor else None
        if self.pretokenize:
            prefix_tokens = self._prefix_tokens(prefix)
            suffix_tokens = self.backend.tokenize(suffix)
            scored = self.backend.score(prefix_tokens + suffix_tokens, ids, delimiters)
            scored["suffix_n"] = len(suffix_tokens)
        else:
            text = prefix + suffix
            bos = self.formatter.server_bos
            if bos and text.startswith(bos):
                # the server adds this one itself; sending it would make two
                text = text[len(bos):]
            scored = self.backend.score(text, ids, delimiters)
            scored["suffix_n"] = 0

        raw = {oid: scored["raw"][i] for oid, i in zip(labels, ids, strict=True)}
        return raw, scored, suffix

    def _resolve_wide(self, count):
        """the wide alphabet and what scoring with it measured, resolved once.

        resolving costs a whole-vocabulary read and verifying costs a scoring
        pass, so both happen on demand and are cached per model.
        """
        if len(self._wide) < count:
            self._wide = derive.ensure_wide_labels(
                self.backend, self.formatter, count, self.cache_dir)
            self._wide_checked = None
        if self._wide_checked is None:
            self._wide_checked = (
                derive.wide_verification(self.formatter, self.cache_dir)
                or derive.verify_wide(self.backend, self.formatter, self._wide))
        return self._wide, self._wide_checked

    def _alphabet(self, count):
        """labels for a list this long, or None for the spec's pinned 52.

        only a question past the ceiling has anything to answer here, so it is
        resolved on demand and never for the common case. returning None below
        the ceiling is what keeps a short list byte-identical to what
        `spec/fixtures/` pins.

        a caller who did not opt in also gets None, so the refusal comes from
        `spec.labels` with its documented message rather than from here.
        """
        # the wide alphabet extends the chat layout's letters; a named layout's
        # model was trained on its own letters and gets no others
        if (count <= len(self._layout["labels"]) or not self.wide_alphabet
                or self.formatter.layout != spec.CHAT):
            return None
        alphabet, checked = self._resolve_wide(count)
        floor = spec.constants()["option_mass_floor_request"]
        mass = checked.get("min_option_mass", 0.0)
        if mass < floor:
            raise ValueError(
                f"{self.backend.model}: a list labelled from this model's own "
                f"vocabulary put option mass {mass:.4g} on its labels, below "
                f"the floor of {floor}. the alphabet resolves and the model "
                f"does not use it, so the readout would be renormalised over "
                f"whatever little landed there. shortlist the options, or "
                f"enable the tournament to accept grouped scoring instead.")
        if checked.get("correct_ratio", 1.0) < 1.0:
            raise ValueError(
                f"{self.backend.model}: answered the verification question "
                f"{checked['smoke_correct']} when it was labelled from this "
                f"model's own vocabulary, having answered the same question "
                f"correctly in A-Za-z. the labels are readable and the model "
                f"cannot choose between them. shortlist the options, or "
                f"enable the tournament to accept grouped scoring instead.")
        self._label_ids.update(alphabet)
        # may be SHORT of `count` on a small tokenizer. returned anyway, so
        # `spec.labels` raises with the message naming the alternatives rather
        # than this returning None and the refusal reading as "not enabled"
        return list(alphabet)[:count]

    def _can_label(self, count):
        """whether one exact pass can label a list this long AND be trusted.

        deliberately does not raise: it is the question the tournament fallback
        asks, and a caller who enabled the tournament wants the approximation
        rather than the refusal.
        """
        if count <= len(self._layout["labels"]):
            return True
        if not self.wide_alphabet or self.formatter.layout != spec.CHAT:
            return False
        alphabet, checked = self._resolve_wide(count)
        floor = spec.constants()["option_mass_floor_request"]
        # both conditions, because measuring showed they come apart: the model
        # with the worst wide-alphabet accuracy of four held the highest wide
        # option mass. see docs/EVALS.md 2a.
        return (len(alphabet) >= count
                and checked.get("min_option_mass", 0.0) >= floor
                and checked.get("correct_ratio", 1.0) >= 1.0)

    def _orders(self, question):
        """option orders to score under, always starting with the declared one.

        two orders means forward and reversed, which is the cheapest pair that
        actually disagrees about position. beyond two, seeded shuffles derived
        from the question itself, so a result reproduces.
        """
        options = list(question.options)
        orders = [options]
        if self.order_averaging > 1:
            orders.append(list(reversed(options)))
        for i in range(2, self.order_averaging):
            seed = hashlib.sha256(
                f"{question.name}:{i}:{[o.id for o in options]}".encode()).digest()
            rng = random.Random(int.from_bytes(seed[:8], "big"))
            shuffled = list(options)
            rng.shuffle(shuffled)
            orders.append(shuffled)
        return orders

    def _prior(self, question, options):
        """what the model answers when the state says nothing at all.

        whatever mass a label holds with no evidence is the model's prejudice
        about that position and wording, not a judgement. dividing it out is
        contextual calibration, and it is cheap here: the null prefix never
        changes, so the server keeps it cached across every request forever.
        """
        key = (question.name, tuple(o.id for o in options))
        if key not in self._prior_cache:
            null_prefix = prompt.render_prefix(self.formatter, "")
            raw, _, _ = self._score_options(null_prefix, question, options)
            probs, mass = extract.renormalise(raw)
            self._prior_cache[key] = probs if mass > 0 else {}
        return self._prior_cache[key]

    def _prefix_tokens(self, prefix):
        """the state is the cached prefix and does not change between the
        questions of one request, so it is tokenised at most once."""
        if self._prefix_cache[0] != prefix:
            self._prefix_cache = (prefix, self.backend.tokenize(prefix))
        return self._prefix_cache[1]

    def _run_tournament(self, prefix, question, group_size):
        """group, score each group, then run off the group winners.

        an option's reported probability is its within-group probability times
        its group winner's probability in the runoff. that is a hierarchical
        approximation, not a distribution read from one position, and it is
        sensitive to how the options were partitioned. it does sum to one.
        """
        options = list(question.options)
        groups = [options[i:i + group_size] for i in range(0, len(options), group_size)]

        within, winners, worst_mass, passes = [], [], 1.0, 0
        total = 0
        for group in groups:
            raw, scored, _ = self._score_options(prefix, question, group)
            total += scored["prompt_total"]
            probs, mass = extract.renormalise(raw)
            passes += 1
            worst_mass = min(worst_mass, mass)
            within.append(probs)
            best = max(probs, key=probs.get) if mass > 0 else group[0].id
            winners.append(next(o for o in group if o.id == best))

        if len(winners) > group_size:
            runoff = self._run_tournament(
                prefix, dataclasses.replace(question, options=tuple(winners)),
                group_size)
            runoff_probs, passes = runoff["probs"], passes + runoff["passes"]
            worst_mass = min(worst_mass, runoff["option_mass"])
            total += runoff["prompt_total"]
        else:
            raw, scored, _ = self._score_options(prefix, question, winners)
            runoff_probs, mass = extract.renormalise(raw)
            total += scored["prompt_total"]
            passes += 1
            worst_mass = min(worst_mass, mass)

        final = {}
        for probs, winner in zip(within, winners, strict=True):
            weight = runoff_probs.get(winner.id, 0.0)
            for oid, p in probs.items():
                final[oid] = p * weight
        return {"probs": final, "option_mass": worst_mass, "passes": passes,
                "prompt_total": total}

    def label_ids(self, labels):
        """label token ids, cached: they depend only on the model."""
        missing = [ell for ell in labels if ell not in self._label_ids]
        for ell in missing:
            tok = self.backend.tokenize(ell)
            if len(tok) != 1:
                raise ValueError(
                    f"label {ell!r} is {len(tok)} tokens on this model; the readout "
                    f"needs one token per label")
            self._label_ids[ell] = tok[0]
        return [self._label_ids[ell] for ell in labels]

    def _decide_jointly(self, state, questions):
        """every question in one prompt, read once and decided together by a
        joint head (SPEC 5.5). order averaging and prior correction do not
        apply: the options are not labelled and have no position to bias."""
        tokens, fields = joint_prompt.encode(self._tokens_of, self.formatter.layout,
                                             state, questions)
        read = self.backend.hidden_states(tokens)
        t0 = time.monotonic()
        rows = self.head.probabilities(
            self.head.logits(read["hidden"], tokens, [f.spans for f in fields]))
        head_ms = (time.monotonic() - t0) * 1000.0
        floor, answers = spec.constants()["option_mass_floor_request"], {}
        for field, row in zip(fields, rows, strict=True):
            question = types.Question(
                field.name, field.kind, "", tuple(types.Option(o, "") for o in field.option_ids),
                legend=(types.legend_of(questions[field.name]["criteria"])
                        if field.kind == types.SCORE else None))
            a = extract.answer(question, dict(zip(field.option_ids, row, strict=True)), floor,
                               measured=False)
            a.update(truncated=False,
                     timing={"prefill_ms": round(read["score_ms"], 2),
                             "score_ms": round(head_ms, 2)},
                     provenance={
                         "model": self.backend.model, "formatter": self.formatter.model,
                         "template_sha256": self.formatter.template_sha256,
                         "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                         "spec_version": spec.constants()["spec_version"],
                         "backend": "http", "readout": "joint-head"})
            answers[field.name] = a
        return {"answers": {name: answers[name] for name in questions},
                "usage": {"input_tokens": len(tokens), "output_tokens": 0}}

    def decide(self, state, questions):
        """answer every question against one shared, prefilled state."""
        if self.head and self.head.joint:
            return self._decide_jointly(state, questions)
        parsed = types.parse_questions(questions, self.formatter.layout)
        floor = spec.constants()["option_mass_floor_request"]

        t0 = time.monotonic()
        prefix = prompt.render_prefix(self.formatter, state)
        prefill_ms = (time.monotonic() - t0) * 1000.0

        answers, usage = {}, {"input_tokens": 0, "output_tokens": 0}
        ceiling = self._layout["max_options_single_pass"]
        group_size = self._layout["max_options_plain_alphabet"]

        for question in sorted(parsed, key=suffix_size):
            t1 = time.monotonic()
            # the exact readout wins when both are enabled: the tournament
            # exists because the alphabet ran out, and it no longer has. it
            # stays as the fallback for a tokenizer that cannot reach the
            # count, since a caller who enabled it named the approximation as
            # acceptable and refusing outright gives them neither.
            if (len(question.options) > ceiling and self.tournament
                    and not self._can_label(len(question.options))):
                run = self._run_tournament(prefix, question, group_size)
                a = extract.answer(question, run["probs"], floor)
                # the distribution is already normalised across groups, so the
                # renormalisation inside answer() cannot see the real mass
                a["option_mass"] = run["option_mass"]
                a["approximate"] = True
                a["flags"] = [*a["flags"], "tournament"]
                if run["option_mass"] < floor:
                    a["flags"].append("low_option_mass")
                scored = {"truncated": False, "score_ms": (time.monotonic() - t1) * 1000.0}
                suffix = f"<tournament over {len(question.options)} options>"
                usage["input_tokens"] += run["prompt_total"]
            else:
                orders = self._orders(question)
                totals = {o.id: 0.0 for o in question.options}
                passes = 0
                for order in orders:
                    raw, scored, suffix = self._score_options(prefix, question, order)
                    for oid, p in raw.items():
                        totals[oid] += p
                    usage["input_tokens"] += scored["prompt_total"]
                    passes += 1
                raw = {oid: p / len(orders) for oid, p in totals.items()}

                # option mass is a property of the readout and must be measured
                # before any debiasing rescales it
                mass = sum(raw.values())
                if self.prior_correction:
                    prior = self._prior(question, question.options)
                    raw = {oid: (p / prior[oid] if prior.get(oid) else p)
                           for oid, p in raw.items()}

                a = extract.answer(question, raw, floor, option_mass=mass,
                                   measured=not self.head)
                a["flags"] += prompt.alphabet_flags(question, self.formatter.layout)
                if len(orders) > 1:
                    a["flags"].append("order_averaged")
                if self.prior_correction:
                    a["flags"].append("prior_corrected")
                a["cache"] = {"prompt_n": scored["prompt_n"],
                              "tokens_cached": scored["tokens_cached"],
                              "retries": scored["retries"], "passes": passes}

            if scored["truncated"]:
                a["flags"].append("truncated")
            a["truncated"] = scored["truncated"]
            a["timing"] = {"prefill_ms": round(prefill_ms, 2),
                           "score_ms": round(scored["score_ms"], 2)}
            a["provenance"] = {
                "model": self.backend.model,
                "formatter": self.formatter.model,
                "template_sha256": self.formatter.template_sha256,
                "prompt_sha256": prompt.prompt_sha256(prefix, suffix),
                "spec_version": spec.constants()["spec_version"],
                "backend": "http",
                "readout": "head" if self.head else "logits",
                "order_averaging": self.order_averaging,
                "prior_correction": self.prior_correction,
            }
            if a.get("approximate"):
                a["provenance"]["scoring"] = "tournament"
            answers[question.name] = a
            # the state is prefilled once; later questions only pay for their
            # own suffix, so charging them the prefill again would misreport it
            prefill_ms = 0.0

        # scored shortest first, returned in the order the caller asked
        return {"answers": {q.name: answers[q.name] for q in parsed}, "usage": usage}


def suffix_size(question):
    """roughly how long a question's suffix renders, to score short ones first.

    a sliding-window model keeps only the last window of a cached prompt, so it
    can reuse the shared state only when the previous prompt's text after that
    state fits the window. a long option list scored between two questions over
    one state makes the second prefill the state again; scored last, it costs
    nothing more. the answers do not depend on the order, since no question
    sees another's answer.
    """
    return len(question.instructions) + sum(len(o.description) + 4 for o in question.options)


def gate(answer, threshold, assume_uncalibrated=False):
    """act or escalate.

    refuses to answer on uncalibrated probabilities unless the caller says so
    explicitly. this is deliberate friction: one measured model reports a top
    probability of 1.000 on answers that are wrong, so a threshold fitted on a
    better behaved model passes everything it emits.
    """
    if not answer.get("calibrated") and not assume_uncalibrated:
        raise ValueError(
            "gating on uncalibrated probabilities. fit a calibration on your own "
            "labelled data, or pass assume_uncalibrated=True to accept that "
            "confidence is not comparable across models.")
    if "low_option_mass" in answer.get("flags", ()):
        return "escalate"
    return "act" if answer["confidence"] >= threshold else "escalate"
