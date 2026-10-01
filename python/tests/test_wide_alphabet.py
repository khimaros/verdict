"""scoring an option list past the pinned 52 labels in one pass.

`A`-`Za`-`z` runs out at 52 and the documented fallback is the tournament,
whose probabilities are a hierarchical approximation sensitive to how the
options happened to be partitioned. a model's own vocabulary carries thousands
of single character letters -- of the first 160 resolved on qwen3.5-9b, 108
are outside ascii -- so a longer list can be labelled exactly and read from one
position instead.

the load bearing property here is that the extension is INVISIBLE below 52. the
same question must produce the same prompt bytes whether or not a wide alphabet
is available, or every fixture in `spec/fixtures/` is describing a prompt the
client no longer sends.
"""

import string

import pytest

from llama_verdict import derive, prompt, spec
from llama_verdict.decide import Decider
from llama_verdict.types import Formatter

TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# single character tokens outside ascii, which is where a long list has to go.
# greek and cyrillic are alphabetic, hiragana is alphabetic, and the punctuation
# is here to be REJECTED rather than used.
GREEK = "".join(chr(0x3B1 + i) for i in range(25))
CYRILLIC = "".join(chr(0x410 + i) for i in range(32))
HIRAGANA = "".join(chr(0x3042 + i) for i in range(40))
PUNCT = "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~"


class VocabBackend:
    """a tokenizer and a vocabulary, no weights.

    one token per character, so `tokenize(a + b) == tokenize(a) + tokenize(b)`
    holds by construction and the merge check is exercised on the cases that
    opt out of it rather than being sidestepped everywhere.
    """

    def __init__(self, merges=(), vocab=None, wide_mass=1.0, wide_correct=True):
        self.model = "fake:Q8_0"
        self.vocab_reads = 0
        self.tokenize_calls = 0
        self.scored = []
        # how much probability this model is willing to put on NON-ASCII
        # labels. the whole point of verifying an alphabet is that a model can
        # be perfectly steerable on A-Za-z and put its mass elsewhere entirely
        # when handed characters it does not use.
        self.wide_mass = wide_mass
        # and whether it still picks the right one. measured, these come apart:
        # minicpm5-2b holds the HIGHEST wide option mass of four models and
        # answers an unambiguous question wrong at every count.
        self.wide_correct = wide_correct
        # labels that stop being their own token once the opening precedes them
        self.merges = set(merges)
        self.vocab = vocab if vocab is not None else (
            string.ascii_letters + string.digits + PUNCT
            + GREEK + CYRILLIC + HIRAGANA)

    def props(self):
        return {"chat_template": TEMPLATE, "bos_token": "", "eos_token": "",
                "build_info": "fake", "model_alias": "fake"}

    def readout(self):
        return None

    def tokenize(self, text, add_special=False):
        self.tokenize_calls += 1
        if len(text) > 1 and text[-1] in self.merges:
            return [ord(c) for c in text[:-2]] + [-ord(text[-1])]
        return [ord(c) for c in text]

    def score(self, prompt_text, label_ids, delimiters=None):
        self.scored.append((prompt_text, list(label_ids)))
        # ids are ord() of the label, so anything past ascii is a wide label
        wide = all(i > 127 for i in label_ids)
        mass = self.wide_mass if wide else 1.0
        # the verify question puts the right answer first, so putting the mass
        # last is how this fake gets it wrong
        winner = label_ids[-1] if wide and not self.wide_correct else label_ids[0]
        return {"raw": {winner: mass, **{i: 0.0 for i in label_ids if i != winner}},
                "truncated": False, "n_probs_used": 64, "retries": 0,
                "score_ms": 1.0, "prompt_n": 1, "prompt_total": 1,
                "tokens_cached": 0, "top_unconstrained": "A"}

    def _post(self, path, payload):
        self.vocab_reads += 1
        return {"completion_probabilities": [{"top_logprobs": [
            {"id": ord(c), "token": c} for c in self.vocab]}]}, 1.0


def question(n):
    """n options whose ids are positional, so a label can be traced to one."""
    return {"pick": {"type": "choice",
                     "instructions": {"rules": ["Which one?"]},
                     "criteria": {f"o{i:03d}": f"option number {i}"
                                  for i in range(n)}}}


@pytest.fixture
def fitted(tmp_path):
    """a derived formatter over the fake, cached where the test can inspect it.

    the cache directory is threaded through everything on purpose: a default
    of `~/.cache` here would have the suite writing entries for a model that
    does not exist, and one draft of this file did exactly that.
    """
    def build(backend, **kw):
        return derive.build(backend, "fake:Q8_0", cache_dir=str(tmp_path), **kw)
    build.cache_dir = str(tmp_path)
    return build


def test_a_list_past_fiftytwo_is_scored_in_one_pass(fitted):
    """the whole point: one readout at one position, not a bracket."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    decider = Decider(backend, formatter, wide_alphabet=True,
                      cache_dir=fitted.cache_dir)

    # the first decision also resolves and verifies the alphabet, which is a
    # once-per-model cost. what matters is the steady state.
    decider.decide("a page", question(80))
    backend.scored.clear()
    answer = decider.decide("a page", question(80))["answers"]["pick"]

    assert len(backend.scored) == 1, "a single pass reads one position once"
    assert not answer.get("approximate"), "a single pass is exact, not approximate"
    assert "tournament" not in answer["flags"]
    assert len(backend.scored[0][1]) == 80, "every option needs its own label id"
    assert len(set(backend.scored[0][1])) == 80, "labels must be distinct"


def test_the_prompt_is_unchanged_at_the_pinned_ceiling(fitted):
    """below 52 the wide alphabet must not exist. the spec fixtures pin these
    bytes, and an alphabet that reordered them would invalidate all of them
    while still answering plausibly."""
    plain = VocabBackend()
    wide = VocabBackend()
    formatter, _ = fitted(plain)
    formatter_w, _ = fitted(wide)

    for n in (1, 26, 27, 52):
        plain.scored.clear()
        wide.scored.clear()
        Decider(plain, formatter).decide("a page", question(n))
        Decider(wide, formatter_w, wide_alphabet=True,
                cache_dir=fitted.cache_dir).decide("a page", question(n))
        assert plain.scored[0] == wide.scored[0], f"{n} options diverged"

    assert wide.vocab_reads == 0, (
        "resolving a vocabulary costs about 45 s on a real tokenizer and no "
        "question at or below the ceiling may pay it")


def test_a_list_past_fiftytwo_is_flagged(fitted):
    """the labels are verified single tokens and nothing more, and the flag is
    how a caller learns which alphabet answered. it earns its place: measured
    with list length held fixed, unfamiliar labels take minicpm5-2b from 16/16
    to 9/16 and drop granite's option mass under the request floor."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    answer = Decider(backend, formatter, wide_alphabet=True,
                     cache_dir=fitted.cache_dir).decide(
        "a page", question(60))["answers"]["pick"]
    assert "extended_alphabet" in answer["flags"]


def test_without_the_wide_alphabet_a_long_list_is_still_refused(fitted):
    """opting in is the contract. a caller who did not ask gets the documented
    refusal naming the alternatives, not a silently different encoding."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    with pytest.raises(ValueError) as e:
        Decider(backend, formatter).decide("a page", question(53))
    assert "tournament" in str(e.value)


def test_the_pinned_letters_come_first(fitted):
    """a-z before anything the vocabulary happened to order earlier, so the
    first 52 labels of a long list are the same 52 a short list would use."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    labels = derive.ensure_wide_labels(backend, formatter, 80, fitted.cache_dir)
    assert list(labels)[:52] == list(spec.constants()["labels"])


def test_the_alphabet_stays_letters_only(fitted):
    """spec section 10: the alphabet is letters. running out of ascii does not
    change that. `)` is the option line delimiter and a digit reads as part of
    the option text, so neither may become a label."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    labels = derive.ensure_wide_labels(backend, formatter, 120, fitted.cache_dir)
    assert all(ell.isalpha() for ell in labels), \
        [ell for ell in labels if not ell.isalpha()]


def test_a_label_that_merges_with_the_opening_is_skipped(fitted):
    """being one token in isolation is not the property being relied on. the
    label is read at the position right after the assistant opening, and a
    candidate that merges there is not readable at all."""
    bad = GREEK[:5]
    backend = VocabBackend(merges=bad)
    formatter, _ = fitted(backend)
    labels = derive.ensure_wide_labels(backend, formatter, 80, fitted.cache_dir)
    assert not set(labels) & set(bad), "a merging candidate reached the alphabet"
    assert len(labels) >= 80


def test_a_candidate_costs_one_round_trip(fitted):
    """the vocabulary read already said which id a character is. asking the
    tokenizer again is a round trip for an answer already in hand, and the only
    question it cannot answer is the one that matters -- whether the character
    survives the assistant opening. measured at 160 labels, that is 160 calls
    against 320, on top of the vocabulary read itself.
    """
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    backend.tokenize_calls = 0
    labels = derive.ensure_wide_labels(backend, formatter, 80, fitted.cache_dir)
    # one per candidate for the merge check, plus one for the opening itself
    assert backend.tokenize_calls <= len(labels) + 1, (
        f"{backend.tokenize_calls} tokenize calls for {len(labels)} labels")


def test_the_scored_id_comes_from_the_vocabulary(fitted):
    """the id being scored has to be the id the model named for that character,
    or the readout is at a position nobody checked."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    ids = derive.ensure_wide_labels(backend, formatter, 80, fitted.cache_dir)
    assert all(ids[c] == ord(c) for c in ids)


def test_the_vocabulary_is_read_once_per_model(fitted, tmp_path):
    """the read is a whole-vocabulary completion, about 45 s on a 248k piece
    tokenizer. it belongs in the formatter's cache entry beside the affixes."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    derive.ensure_wide_labels(backend, formatter, 80, fitted.cache_dir)
    assert backend.vocab_reads == 1

    again, cached = derive.build(backend, "fake:Q8_0", cache_dir=str(tmp_path))
    assert cached, "the formatter should have come from the cache"
    derive.ensure_wide_labels(backend, again, 80, fitted.cache_dir)
    assert backend.vocab_reads == 1, "a second process re-read the vocabulary"


def test_a_model_with_too_few_labels_names_the_alternative(fitted):
    """a small tokenizer cannot always reach the count asked for, and the
    caller needs to be told what to do rather than handed 60 labels for 80
    options."""
    backend = VocabBackend(vocab=string.ascii_letters + PUNCT)
    formatter, _ = fitted(backend)
    with pytest.raises(ValueError) as e:
        Decider(backend, formatter, wide_alphabet=True,
                cache_dir=fitted.cache_dir).decide("a page", question(80))
    assert "tournament" in str(e.value)


def test_the_tournament_still_catches_a_model_with_too_few_labels(fitted):
    """a caller who enabled BOTH asked for the exact readout and separately
    named the approximation as acceptable. refusing outright hands them neither
    when they told us which one they would take."""
    backend = VocabBackend(vocab=string.ascii_letters + PUNCT)
    formatter, _ = fitted(backend)
    answer = Decider(backend, formatter, wide_alphabet=True, tournament=True,
                     cache_dir=fitted.cache_dir).decide(
        "a page", question(80))["answers"]["pick"]
    assert answer["approximate"], "an approximation must say so"
    assert "tournament" in answer["flags"]


def test_the_exact_path_wins_when_both_are_enabled(fitted):
    """the tournament exists because the alphabet ran out. where it has not,
    the approximation has nothing to offer."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    answer = Decider(backend, formatter, wide_alphabet=True, tournament=True,
                     cache_dir=fitted.cache_dir).decide(
        "a page", question(80))["answers"]["pick"]
    assert not answer.get("approximate")
    assert "tournament" not in answer["flags"]


# a wide alphabet is verified the way a formatter is: by SCORING with it and
# reading the option mass, not by predicting usability from properties of the
# characters. two such predictions were measured and both failed -- dropping
# lookalikes made qwen3.5-9b worse, and emission base rate ranks the failing
# group above the passing one. docs/EVALS.md 2a.

def test_a_resolved_wide_alphabet_is_verified_by_scoring_with_it(fitted):
    """the check has to use a list labelled ENTIRELY with wide labels. at 104
    options half the labels are ascii, so total option mass would stay healthy
    on the pinned half and say nothing about the other."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    derive.ensure_wide_labels(backend, formatter, 80, fitted.cache_dir)

    checked = derive.wide_verification(formatter, fitted.cache_dir)
    assert checked, "resolving an alphabet must record what it measured"
    assert checked["min_option_mass"] > 0.9
    assert all(not ell.isascii() for ell in checked["labels"]), checked["labels"]
    # several counts, as the formatter check does. a short list is the easiest
    # case a model ever sees and passing it proves the least.
    assert len(checked["option_counts"]) > 1, checked["option_counts"]
    assert checked["smoke_correct"]


def test_an_unusable_wide_alphabet_is_refused(fitted):
    """a model that will not put probability on these characters must be told
    so, not served a readout whose mass lives somewhere else entirely."""
    backend = VocabBackend(wide_mass=0.31)
    formatter, _ = fitted(backend)
    with pytest.raises(ValueError) as e:
        Decider(backend, formatter, wide_alphabet=True,
                cache_dir=fitted.cache_dir).decide("a page", question(80))
    assert "option mass" in str(e.value)


def test_an_unusable_wide_alphabet_falls_back_to_the_tournament(fitted):
    """the approximation is worth more than a refusal to a caller who named it
    as acceptable, and worth more than an exact read nobody should trust."""
    backend = VocabBackend(wide_mass=0.31)
    formatter, _ = fitted(backend)
    answer = Decider(backend, formatter, wide_alphabet=True, tournament=True,
                     cache_dir=fitted.cache_dir).decide(
        "a page", question(80))["answers"]["pick"]
    assert answer["approximate"]
    assert "tournament" in answer["flags"]


def test_an_alphabet_with_healthy_mass_and_wrong_answers_is_refused(fitted):
    """mass alone is not the check, and measuring said so. across four models
    the WORST wide-alphabet accuracy came with the HIGHEST wide option mass --
    minicpm5-2b at 0.9658, answering an unambiguous question wrong at every
    count while scoring 3/3 on the same question in ascii."""
    backend = VocabBackend(wide_mass=0.97, wide_correct=False)
    formatter, _ = fitted(backend)
    with pytest.raises(ValueError) as e:
        Decider(backend, formatter, wide_alphabet=True,
                cache_dir=fitted.cache_dir).decide("a page", question(80))
    assert "0/" in str(e.value), str(e.value)


def test_a_usable_wide_alphabet_is_served(fitted):
    backend = VocabBackend(wide_mass=0.95)
    formatter, _ = fitted(backend)
    answer = Decider(backend, formatter, wide_alphabet=True,
                     cache_dir=fitted.cache_dir).decide(
        "a page", question(80))["answers"]["pick"]
    assert not answer.get("approximate")
    assert "extended_alphabet" in answer["flags"]


def test_the_verification_is_cached_with_the_alphabet(fitted, tmp_path):
    """it costs a scoring pass, and it answers a per-model question."""
    backend = VocabBackend()
    formatter, _ = fitted(backend)
    derive.ensure_wide_labels(backend, formatter, 80, fitted.cache_dir)
    before = len(backend.scored)

    again, _ = derive.build(backend, "fake:Q8_0", cache_dir=str(tmp_path))
    derive.ensure_wide_labels(backend, again, 80, fitted.cache_dir)
    assert len(backend.scored) == before, "the alphabet was re-verified"
    assert derive.wide_verification(again, fitted.cache_dir)


def test_labels_accepts_an_explicit_alphabet():
    """`spec.labels` is what every implementation calls, so the extension point
    is here rather than in the python client."""
    alphabet = list(spec.constants()["labels"]) + list(GREEK)
    assert spec.labels(60, alphabet)[:52] == list(spec.constants()["labels"])
    assert spec.labels(60, alphabet)[52] == GREEK[0]


def test_the_suffix_renders_the_alphabet_it_was_given():
    """the label column in the prompt has to agree with the ids being scored,
    and they are produced by two different functions."""
    from llama_verdict import types
    q = types.parse_question("pick", question(60)["pick"])
    alphabet = list(spec.constants()["labels"]) + list(GREEK)
    reference = Formatter(model="r", system_open="", system_close="",
                          user_open="", user_close="", assistant_open="A:")
    rendered = prompt.render_suffix(reference, q, alphabet)
    mapped = prompt.label_map(q, alphabet)
    assert f"\n{GREEK[0]}) option number 52" in rendered
    assert mapped["o052"] == GREEK[0]
