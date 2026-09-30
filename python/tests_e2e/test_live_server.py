"""end to end against a live llama-server.

gated on LLAMA_VERDICT_URL so the suite stays runnable without one; the
environment and the .env at the repo root both fill it:

  make test-e2e   # with the backend in .env, or exported in the shell
"""

import glob
import os

import pytest

from llama_verdict import config
from llama_verdict.backend import HttpBackend
from llama_verdict.decide import Decider
from llama_verdict.types import Formatter

URL = config.get("LLAMA_VERDICT_URL")
MODEL = config.get("LLAMA_VERDICT_MODEL")
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spec")

pytestmark = pytest.mark.skipif(not URL, reason="set LLAMA_VERDICT_URL")

TICKET = "My invoice charged me twice for the same month and I want a refund."
QUESTIONS = {
    "queue": {"type": "choice", "instructions": "Which queue should handle this?",
              "criteria": {"access": "Account access support.",
                           "billing": "Billing support.",
                           "technical": "Technical support."}},
    "refund": {"type": "boolean", "instructions": "Do they want money back?"},
}


def formatter_path():
    name = (MODEL or "").replace(":", "_").replace("/", "_") + ".json"
    path = os.path.join(SPEC, "formatters", name)
    if not os.path.exists(path):
        pytest.skip(f"no pinned formatter for {MODEL}; run scripts/derive_formatter.py")
    return path


@pytest.fixture(scope="module")
def formatter():
    return Formatter.load(formatter_path())


def decider(formatter, **kw):
    return Decider(HttpBackend(URL, MODEL), formatter, **kw)


def test_runtime_derivation_reproduces_the_pinned_formatter(formatter, tmp_path):
    """spec 4.4: pinned formatters are golden regression fixtures, so deriving
    one today must reproduce it byte for byte. a difference here means the
    engine, the derivation or the model's template moved, and it should be a
    diff rather than a silent change in behaviour."""
    from llama_verdict import derive

    fresh, cached = derive.build(HttpBackend(URL, MODEL), MODEL,
                                 cache_dir=str(tmp_path))
    assert not cached, "a fresh cache directory should force derivation"
    for field in ("system_open", "system_close", "user_open", "user_close",
                  "assistant_open"):
        assert getattr(fresh, field) == getattr(formatter, field), field
    assert fresh.template_sha256 == formatter.template_sha256
    assert fresh.label_ids == formatter.label_ids


def test_derivation_caches_by_template_sha256(tmp_path):
    """spec 4.3: once per model, not once per process."""
    from llama_verdict import derive

    backend = HttpBackend(URL, MODEL)
    _, first = derive.build(backend, MODEL, cache_dir=str(tmp_path))
    second_formatter, second = derive.build(backend, MODEL, cache_dir=str(tmp_path))
    assert (first, second) == (False, True)
    assert second_formatter.verification["mean_option_mass"] > 0.9


def test_decide_returns_healthy_option_mass(formatter):
    result = decider(formatter).decide(TICKET, QUESTIONS)
    for name, a in result["answers"].items():
        assert a["option_mass"] > 0.9, f"{name} option mass {a['option_mass']}"
        assert "low_option_mass" not in a["flags"]
        assert abs(sum(a["probs"].values()) - 1.0) < 1e-6
    assert result["answers"]["queue"]["top"] == "billing"
    assert result["usage"]["output_tokens"] == 0


def test_text_and_token_prompts_agree(formatter):
    """sending the prompt as text costs no /tokenize round trip and must be the
    same token sequence. if this ever fails, the spec's boundary assertion has
    stopped holding for this model and the text path is no longer safe."""
    as_text = decider(formatter, pretokenize=False).decide(TICKET, QUESTIONS)
    as_tokens = decider(formatter, pretokenize=True).decide(TICKET, QUESTIONS)

    for name in QUESTIONS:
        text, tokens = as_text["answers"][name], as_tokens["answers"][name]
        assert text["top"] == tokens["top"]
        for option_id, p in text["probs"].items():
            # batch shape moves the last digits; argmax must not move at all
            assert abs(p - tokens["probs"][option_id]) < 1e-3, name


def test_split_tokenisation_matches_joint(formatter):
    """the assertion the text path depends on, checked directly."""
    from llama_verdict import prompt, types
    backend = HttpBackend(URL, MODEL)
    question = types.parse_questions(QUESTIONS)[0]
    prefix = prompt.render_prefix(formatter, TICKET)
    suffix = prompt.render_suffix(formatter, question)
    assert (backend.tokenize(prefix) + backend.tokenize(suffix)
            == backend.tokenize(prefix + suffix))


def test_pinned_label_ids_match_the_live_model(formatter):
    """the formatter pins label token ids to avoid a round trip per label.
    a pinned id that no longer matches the model would silently score the
    wrong token."""
    backend = HttpBackend(URL, MODEL)
    for label, pinned in list(formatter.label_ids.items())[:8]:
        assert backend.tokenize(label) == [pinned], label


def test_the_wide_alphabet_labels_a_long_list_against_a_real_tokenizer():
    """spec 10.1 end to end. the offline suite drives a tokenizer written for
    it, which cannot answer the question the whole path rests on: does a REAL
    vocabulary yield enough single-character letters that survive the assistant
    opening?

    derives rather than loading a pinned formatter, because the wide alphabet
    is cached in the derived entry beside the affixes.
    """
    from llama_verdict import derive
    backend = HttpBackend(URL, MODEL)
    built, _ = derive.build(backend, MODEL)
    decide = Decider(backend, built, wide_alphabet=True)

    n = 80
    criteria = {f"o{i:03d}": f"Page element number {i}." for i in range(n - 1)}
    criteria["target"] = "The departure date field, which opens a date picker."
    answer = decide.decide(
        "Page: a flight search form.",
        {"pick": {"type": "choice",
                  "instructions": "Which element opens the date picker?",
                  "criteria": criteria}})["answers"]["pick"]

    assert not answer.get("approximate"), "a wide-alphabet read is exact"
    assert "extended_alphabet" in answer["flags"]
    assert len(answer["probs"]) == n, "every option needs its own label"
    # deliberately NOT an accuracy assertion. docs/EVALS.md 2a measures that,
    # and a small model legitimately gets an 80 option question wrong. what has
    # to hold here is that the readout landed on the declared labels at all.
    assert answer["option_mass"] > 0.5, answer["option_mass"]


@pytest.mark.parametrize(
    "path", sorted(glob.glob(os.path.join(SPEC, "fixtures", "*.json"))))
def test_every_fixture_scores_without_error(path, formatter):
    """the fixtures exercise the prompt builder; this checks a live model will
    actually answer them, which the offline suite cannot."""
    import json
    with open(path) as f:
        fixture = json.load(f)
    questions = fixture["questions"]
    if any(len(q.get("criteria") or {}) > 52 for q in questions.values()):
        pytest.skip("needs the opt-in tournament or wide alphabet path")
    result = decider(formatter).decide(fixture["state"], questions)
    for name, a in result["answers"].items():
        assert a["option_mass"] > 0.5, f"{fixture['id']}/{name}"
