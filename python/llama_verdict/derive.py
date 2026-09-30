"""recover a model's affixes from the chat template it already carries.

spec/SPEC.md section 4.3: the formatter is derived when a model is first used,
not shipped per model. a pinned artifact can only cover models someone thought
to pin, which fails the case that matters most -- a user pointing the library
at an arbitrary gguf. `spec/formatters/` stays as golden regression fixtures
(section 4.4), not as the runtime source.

jinja2 is imported inside `environment()` on purpose. the reference client is
stdlib only, and a model whose formatter is already pinned or cached never
renders a template, so the dependency is paid only by the path that needs it.
"""

import hashlib
import json
import os
import re

from . import spec
from .types import Formatter

# sentinels must survive `| trim` and must not collide with template markup
SYS_SENTINEL = "QQSYSTEMQQ"
USER_SENTINEL = "QQUSERQQ"

# a tag pair, never a `<|special|>` token: those are not pairs and closing one
# would be nonsense
TAG = re.compile(r"<([a-z][a-z0-9_]*)>")

CACHE_HOME = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
CACHE_DIR = os.path.join(CACHE_HOME, "verdict", "formatters")

# derivation that cannot clear the mass floor here has not found the opening,
# whatever the affixes look like.
#
# checked at SEVERAL OPTION COUNTS, not one. a two-option question is the
# easiest case a model ever sees and passing it proves less than it appears:
# lfm2.5-2.6b clears two options at 0.9569 and collapses to 0.1535 by 52, so a
# single small case admitted a formatter that cannot be trusted on any real
# question. the counts bracket what callers actually send.
VERIFY_STATE = "The sky on a clear day is blue."
VERIFY_COUNTS = (2, 26, 52)
# the wide alphabet is checked at several counts, every label drawn from it.
# one 26 option list was the first design and it admitted the model with the
# worst measured wide-alphabet accuracy, which is the same mistake the
# single two-option formatter check made.
WIDE_VERIFY_COUNTS = (8, 26, 52)
VERIFY_FILLER = [
    "the sky is green", "the sky is orange", "the sky is violet",
    "the sky is crimson", "the sky is amber", "the sky is teal",
    "the sky is magenta", "the sky is olive", "the sky is indigo",
    "the sky is scarlet", "the sky is umber", "the sky is ochre",
]


def verify_question(count):
    """one unambiguous question padded to `count` options.

    the answer stays first so the check measures MASS rather than the position
    bias measured separately in docs/EVALS.md -- a formatter check that also
    failed on a weak model's blind spot would refuse a usable formatter.
    """
    criteria = {"blue": "the sky is blue"}
    for i in range(count - 1):
        criteria[f"d{i:02d}"] = VERIFY_FILLER[i % len(VERIFY_FILLER)]
    return {"colour": {"type": "choice", "criteria": criteria,
                       "instructions": {"rules": ["What colour is the sky?"]}}}


def environment():
    """the transformers-style jinja environment a chat template expects.

    templates call raise_exception and strftime_now, and some use
    {% generation %}, which is a transformers extension rather than jinja.
    """
    import datetime
    import typing

    import jinja2
    from jinja2 import nodes
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class Generation(Extension):
        tags: typing.ClassVar = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.CallBlock(self.call_method("_noop", []), [], [],
                                   body).set_lineno(lineno)

        def _noop(self, caller):
            return caller()

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def strftime_now(fmt):
        return datetime.datetime.now(tz=datetime.timezone.utc).strftime(fmt)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols", Generation])
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


def render(tpl, messages, gen_prompt, bos="", eos=""):
    return tpl.render(messages=messages, add_generation_prompt=gen_prompt,
                      enable_thinking=False, tools=None,
                      bos_token=bos, eos_token=eos)


def affixes_from_template(template_text, bos="", eos="", allow_empty_opening=False):
    """recover each affix by diffing renders that differ by one thing.

    with-system vs without-system isolates where the system turn ends and the
    user turn begins; with-generation-prompt vs without isolates the assistant
    opening. the no-system render's opening is kept as `bare_user_open`, for a
    layout whose model is prompted with no system turn.

    an empty assistant opening means the label would follow the user turn
    directly. that is refused for the chat layout, where it has meant a
    derivation that went wrong, and allowed for a layout that says its model
    answers there.
    """
    tpl = environment().from_string(template_text)
    sys_msg = {"role": "system", "content": SYS_SENTINEL}
    usr_msg = {"role": "user", "content": USER_SENTINEL}

    full = render(tpl, [sys_msg, usr_msg], True, bos, eos)
    no_gen = render(tpl, [sys_msg, usr_msg], False, bos, eos)
    no_sys = render(tpl, [usr_msg], True, bos, eos)

    for name, text in (("full", full), ("no_gen", no_gen)):
        if SYS_SENTINEL not in text or USER_SENTINEL not in text:
            raise ValueError(f"template dropped a sentinel in the {name} render")

    i_s, i_u = full.index(SYS_SENTINEL), full.index(USER_SENTINEL)
    if i_u < i_s:
        raise ValueError("template reordered the system turn after the user turn")

    system_open = full[:i_s]
    between = full[i_s + len(SYS_SENTINEL):i_u]
    after = full[i_u + len(USER_SENTINEL):]

    # `between` is system_close followed by user_open. the no-system render
    # opens the user turn the same way but carries any bos at the very front,
    # so the split point is their longest common suffix.
    lead = no_sys[:no_sys.index(USER_SENTINEL)]
    k = next((n for n in range(min(len(between), len(lead)), 0, -1)
              if between[-n:] == lead[-n:]), 0)
    if not k:
        raise ValueError("could not locate where the user turn opens")
    user_open, system_close = between[-k:], between[:-k]

    user_close = no_gen[no_gen.index(USER_SENTINEL) + len(USER_SENTINEL):]
    if not after.startswith(user_close):
        raise ValueError("generation prompt did not extend the ungenerated render")
    assistant_open = after[len(user_close):]

    if not assistant_open and not allow_empty_opening:
        raise ValueError("template produced an empty assistant opening")

    return {"system_open": system_open, "system_close": system_close,
            "user_open": user_open, "user_close": user_close,
            "assistant_open": assistant_open, "bare_user_open": lead}


def close_open_blocks(opening):
    """close any tag the generation prompt opened and did not close.

    a template that ends mid-block puts the label position inside that block,
    and the model goes on writing the block instead of answering. lfm2.5-2.6b
    hardcodes `<|im_start|>assistant\\n<think>` with no enable_thinking branch,
    reads 5.7e-07, and reads 0.9582 once the block is closed. qwen3.5 does the
    same thing correctly, emitting `<think>\\n\\n</think>\\n\\n` itself.

    this is a property of the string rather than knowledge about a model, so
    there is no per-model table: a tag is `<word>` with no matching `</word>`.
    the `<|...|>` special-token forms are deliberately not matched -- they are
    not tag pairs and closing one would be nonsense.

    NOT a general repair. it fixes an unclosed block and nothing else; a
    generation prompt that stops before content for some other reason, as
    harmony's does, still needs an explicit opening.
    """
    for tag in dict.fromkeys(TAG.findall(opening)):
        if f"</{tag}>" not in opening:
            opening += f"\n\n</{tag}>\n\n"
    return opening


def vocabulary(backend, prompt):
    """every token the model can emit, as (id, text), from one scoring call.

    the n_probs ladder tops out at the whole vocabulary, and each entry already
    carries its id and its text. so the model can be asked to enumerate its own
    alphabet instead of being matched against a list of blocks someone thought
    to write down -- the same argument that moved formatters to runtime.

    one call, and on a 248k-piece vocabulary it takes about 45 seconds, so it
    is worth doing only when a question needs more labels than ascii provides.
    """
    out, _ = backend._post("/completion", {
        "prompt": prompt, "n_predict": 1,
        "n_probs": spec.constants()["n_probs_ladder"][-1],
        "cache_prompt": True, "post_sampling_probs": False, "temperature": -1.0})
    entries = out["completion_probabilities"][0]["top_logprobs"]
    return [(e["id"], e["token"]) for e in entries]


def label_candidates(vocab):
    """single-character LETTERS as (text, id), pinned ascii first, then by id.

    letters only, which is the same rule spec section 10 already states for
    `A`-`Za`-`z`: running out of ascii does not make a digit or a bracket a
    good label. `)` is the option line delimiter, and a digit reads as part of
    the option text it labels.

    ascending id puts accented latin and cyrillic first, which are HOMOGLYPHS
    of the ascii labels -- cyrillic `o` is drawn identically to latin `o` and
    is a different token. that looked like a bug worth filtering and it is
    not: dropping every lookalike takes qwen3.5-9b from 16/16 at mass 0.7780
    to 11/16 at 0.3146. **do not add a confusables filter here.**

    a base-rate filter was the next proposal and is also refused. measured, the
    dropped-lookalike set is emitted MORE readily than the set that scores
    better, so base rate does not separate them and a floor on it would not
    have caught the failing group. it does separate ascii from everything else,
    by about 300x, which is already expressed by putting the pinned 52 first.
    docs/EVALS.md 2a.

    the pinned letters come first so a list of 52 or fewer options is labelled
    byte-identically either way. past that the order has to be deterministic
    and model-intrinsic or two runs would build different prompts, so it is
    token id ascending -- a property of the tokenizer, not of this process.
    ascending id also tends to put the commoner characters first, since a bpe
    merge learned early gets a low id, which is the order worth having if an
    unfamiliar label costs anything.

    the id travels with the text because the vocabulary already answered it:
    an entry whose text is one character IS that single token, so re-asking
    `/tokenize` would be a round trip for something in hand.
    """
    seen, ascii_labels, wide = set(), {}, []
    for tid, text in sorted(vocab):
        if len(text) != 1 or not text.isalpha() or text in seen:
            continue
        seen.add(text)
        if text.isascii():
            ascii_labels[text] = tid
        else:
            wide.append((text, tid))
    pinned = [(ell, ascii_labels[ell]) for ell in spec.constants()["labels"]
              if ell in ascii_labels]
    rest = [(ell, tid) for ell, tid in ascii_labels.items()
            if ell not in set(spec.constants()["labels"])]
    return pinned + rest + wide


def scored_opening(affixes, layout=spec.CHAT):
    """the text the label follows: the assistant opening and any prefill, or
    the end of the user turn where a template opens the answer with nothing."""
    return ((affixes["assistant_open"] + spec.layout(layout)["assistant_prefill"])
            or affixes["user_close"])


def label_ids(backend, affixes, layout=spec.CHAT):
    """every label a single token, and still its own token after the opening.

    pinned into the artifact because they are a per-model constant: resolving
    them at runtime cost 26 of the 29 tokenize calls in a profiled decision.
    """
    opening = scored_opening(affixes, layout)
    open_tok = backend.tokenize(opening)
    ids = {}
    for label in spec.layout(layout)["labels"]:
        token = backend.tokenize(label)
        if len(token) != 1 or backend.tokenize(opening + label) != open_tok + token:
            return None, label
        ids[label] = token[0]
    return ids, None


def wide_label_ids(backend, affixes, count, prompt=None):
    """`count` usable labels, going outside ascii when ascii runs out.

    A-Za-z gives 52 and the spec falls back to a tournament past that, which
    reports approximate probabilities. asking the model for its own single
    character letters gives far more -- of the first 160 resolved on
    qwen3.5-9b, 108 are outside ascii -- so a list that used to need grouping
    can be scored in one pass.

    a candidate still has to survive the merge check: being one token in
    isolation does not mean it stays one token after the opening, and that is
    the position being read. so candidates are verified in order and the walk
    stops as soon as `count` of them have passed, which keeps the common case
    at exactly the cost it is today.

    the merge check is the ONLY round trip per candidate. whether the character
    is a single token is already settled by the vocabulary entry it came from;
    whether it stays one after the opening is not, and cannot be.
    """
    opening = affixes["assistant_open"]
    open_tok = backend.tokenize(opening)
    vocab = vocabulary(backend, prompt or opening)

    ids = {}
    for label, tid in label_candidates(vocab):
        if len(ids) >= count:
            break
        if backend.tokenize(opening + label) != [*open_tok, tid]:
            continue
        ids[label] = tid
    return ids


def verify_wide(backend, formatter, ids, counts=WIDE_VERIFY_COUNTS):
    """score an unambiguous question labelled ENTIRELY from the wide alphabet.

    usability is MEASURED here rather than predicted from what the characters
    look like. three predictions have now failed: dropping characters
    confusable with ascii made qwen3.5-9b worse, emission base rate ranks the
    failing alphabet above the passing one, and option mass on one 26 option
    list admits the model with the WORST wide-alphabet accuracy. docs/EVALS.md
    2a.

    every option carries a wide label on purpose. the pinned 52 come first, so
    any list long enough to REACH the wide labels is also half ascii, and its
    option mass would stay healthy on the familiar half while saying nothing
    about the other.

    several counts for the same reason `verify` uses several: a short list is
    the easiest case a model ever sees, and the wide alphabet degrades with
    length faster than the pinned one does.
    """
    from . import extract, prompt, types

    wide = [ell for ell in ids if not ell.isascii()]
    usable = [n for n in counts if 2 <= n <= len(wide)]
    if not usable:
        return {}

    masses, correct = [], 0
    for count in usable:
        alphabet = wide[:count]
        question = types.parse_question("colour", verify_question(count)["colour"],
                                        formatter.layout)
        labels = prompt.label_map(question, alphabet)
        label_ids = [ids[ell] for ell in labels.values()]
        scored = backend.score(
            prompt.render_prefix(formatter, VERIFY_STATE)
            + prompt.render_suffix(formatter, question, alphabet), label_ids)
        raw = {oid: scored["raw"][i]
               for oid, i in zip(labels, label_ids, strict=True)}
        probs, mass = extract.renormalise(raw)
        masses.append(mass)
        correct += int(bool(probs) and max(probs, key=probs.get) == "blue")

    return {"option_counts": usable, "labels": wide[:max(usable)],
            "min_option_mass": min(masses),
            "mean_option_mass": sum(masses) / len(masses),
            "smoke_correct": f"{correct}/{len(usable)}",
            "correct_ratio": correct / len(usable)}


def wide_verification(formatter, cache_dir=CACHE_DIR):
    """what scoring with this model's wide alphabet actually measured."""
    if formatter.wide_verification:
        return formatter.wide_verification
    cached = os.path.join(cache_dir, f"{formatter.template_sha256}.json")
    if os.path.exists(cached):
        with open(cached) as f:
            return json.load(f).get("wide_verification") or {}
    return {}


def ensure_wide_labels(backend, formatter, count, cache_dir=CACHE_DIR):
    """at least `count` labels for this model, resolved once, verified, cached.

    the vocabulary read behind this is a whole-vocabulary completion -- about
    45 seconds on a 248k piece tokenizer -- so it is written back into the
    formatter's own cache entry and the next process pays nothing. a formatter
    that was pinned from a path has no entry to write to and simply resolves in
    memory.

    returns a label -> token id mapping in label order; python preserves it and
    the caller depends on that, since the order IS the alphabet.
    """
    if len(formatter.wide_label_ids) >= count:
        return dict(formatter.wide_label_ids)

    ids = wide_label_ids(backend, {"assistant_open": formatter.assistant_open},
                         count)
    checked = verify_wide(backend, formatter, ids)
    cached = os.path.join(cache_dir, f"{formatter.template_sha256}.json")
    if os.path.exists(cached):
        with open(cached) as f:
            record = json.load(f)
        record["wide_label_ids"] = ids
        record["wide_verification"] = checked
        with open(cached, "w") as f:
            json.dump(record, f, indent=2)
    return ids


def verify(backend, formatter, ids, counts=VERIFY_COUNTS):
    """score the same unambiguous question at several option counts.

    reports the MINIMUM mass as well as the mean, and the floor is applied to
    the minimum, because a formatter is only as trustworthy as its worst
    realistic case.

    deliberately not via `Decider`: it calls `check_usable`, which refuses a
    formatter carrying no verification, and verification is the step that
    produces it. so this uses the same primitives one layer down.
    """
    from . import extract, prompt, types

    masses, correct = [], 0
    usable = [n for n in counts if n <= len(ids)]
    # a layout with fewer labels is still checked at its own ceiling
    if len(ids) < max(counts) and len(ids) not in usable:
        usable.append(len(ids))
    for count in usable:
        question = types.parse_question("colour", verify_question(count)["colour"],
                                        formatter.layout)
        labels = prompt.label_map(question, layout=formatter.layout)
        label_ids = [ids[label] for label in labels.values()]
        scored = backend.score(
            prompt.render_prefix(formatter, VERIFY_STATE)
            + prompt.render_suffix(formatter, question), label_ids)
        raw = {oid: scored["raw"][i]
               for oid, i in zip(labels, label_ids, strict=True)}
        probs, mass = extract.renormalise(raw)
        masses.append(mass)
        correct += int(bool(probs) and max(probs, key=probs.get) == "blue")

    return {"cases": len(usable), "option_counts": usable,
            "mean_option_mass": sum(masses) / len(masses),
            "min_option_mass": min(masses),
            "smoke_correct": f"{correct}/{len(usable)}",
            "labels_single_token": True}


def layout_for(readout):
    """the layout a registry readout names: `chat`, `layout:<name>` or `head`.

    the registry is the authority on how a model must be read, because that is
    a fact about how the model was trained, which its gguf does not record: a
    fine-tuned base model still carries its base's chat template.
    """
    if readout in (None, "", spec.CHAT):
        return spec.CHAT
    kind, _, name = readout.partition(":")
    if kind == "layout" and name:
        return name
    if kind == "head":
        raise ValueError(
            "the model registry says this model answers through a head of its "
            "own, not through next-token logits. serve it with its own server; "
            "reading llama-server logits would return made-up probabilities.")
    raise ValueError(f"unknown readout {readout!r} in the model registry")


def build(backend, model, floor=None, cache_dir=CACHE_DIR, assistant_open=None,
          layout=None):
    """the formatter for this model, derived on first use and cached by the
    template's sha256 so it is once per model rather than once per process.

    a template whose sha256 is already cached skips straight to the cached
    table, which is what makes this cheap enough to do at startup.

    `layout` names one of spec/layouts for a model trained on a layout of its
    own; left unset, it comes from the registry readout the backend reports.
    """
    constants = spec.constants()
    layout = layout or layout_for(backend.readout())
    floor = spec.layout(layout)["option_mass_floor_formatter"] if floor is None else floor

    props = backend.props()
    template = props["chat_template"]
    # an override or a layout changes the prompt, so it must change the key too
    key = template + (assistant_open or "") + ("" if layout == spec.CHAT else layout)
    digest = hashlib.sha256(key.encode()).hexdigest()

    cached = os.path.join(cache_dir, f"{digest}.json")
    if os.path.exists(cached):
        with open(cached) as f:
            return Formatter.from_dict(json.load(f)), True

    declared = layout != spec.CHAT and spec.load_layout(layout).get("affixes")
    if declared:
        affixes = dict(declared)
    else:
        # a layout that declares no affixes keeps the model's own chat template
        # and changes only what goes inside its turns
        affixes = affixes_from_template(template, props.get("bos_token", ""),
                                        props.get("eos_token", ""),
                                        allow_empty_opening=layout != spec.CHAT)
        if spec.layout(layout)["system_turn"]:
            del affixes["bare_user_open"]
    if assistant_open:
        # a template's generation prompt ends where the template ends, which is
        # not always where content begins. gpt-oss derives '<|start|>assistant'
        # and reads 0.0000 because harmony has the model pick a channel next;
        # the same model reads 0.9909 once the opening names the channel. the
        # derivation is right and the template is right, so the escape hatch is
        # an override rather than a fix.
        affixes = dict(affixes, assistant_open=assistant_open)
    elif not declared:
        affixes = dict(affixes,
                       assistant_open=close_open_blocks(affixes["assistant_open"]))
    ids, bad = label_ids(backend, affixes, layout)
    if not ids:
        raise ValueError(
            f"{model}: label {bad!r} is not a single token after the assistant "
            f"opening {scored_opening(affixes, layout)!r}, so it cannot be scored by "
            f"id. the model may still support fewer labels, or different ones: "
            f"derive.wide_label_ids() asks it which tokens it can use.")

    record = {"spec_version": constants["spec_version"], "model": model,
              "model_alias": props.get("model_alias"),
              "template_sha256": digest, "layout": layout,
              "derived_from": ("llama-server /props chat_template" if layout == spec.CHAT
                               else f"spec/layouts/{layout}.json"),
              "affixes": affixes, "label_ids": ids}
    formatter = Formatter.from_dict(dict(record, verification={}))

    checked = verify(backend, formatter, ids)
    # the floor applies to the WORST case, not the average. lfm2.5-2.6b
    # averages well across counts and reads 0.1535 at 52, and a mean would
    # have admitted it.
    if checked["min_option_mass"] < floor:
        # name the opening. a mass near zero usually means the opening stops
        # short of where content begins rather than that the model cannot be
        # scored: gpt-oss-20b derives '<|start|>assistant' and reads 0.0000,
        # because harmony expects the model to emit '<|channel|>final<|message|>'
        # itself, and the same model reads 0.9909 once the opening includes it.
        raise ValueError(
            f"{model}: option mass {checked['min_option_mass']:.4g} at "
            f"{checked['option_counts']} options is below the floor of "
            f"{floor}. the derived assistant opening is "
            f"{affixes['assistant_open']!r}; if the model's format expects more "
            f"before content begins, pass a formatter that spells it out.")
    record["verification"] = dict(checked, build_info=props.get("build_info"))

    os.makedirs(cache_dir, exist_ok=True)
    with open(cached, "w") as f:
        json.dump(record, f, indent=2)
    return Formatter.from_dict(record), False
