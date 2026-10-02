"""a model read through its decision head, against the weights and the author.

akhilaaa3/Jev-Omni publishes four decisions with the probabilities its own
torch classifier returned for them (`verification.json` @ 5addda8). reading
the gguf through verdict has to land on the same numbers, which is the only
check that the prompt, the hidden state and the head are all the author's:
any one of them wrong still yields a confident distribution.

gated on the backend serving the model in embedding mode; `VERDICT_HEAD_MODEL`
names it when it is not the default id.
"""

import http.client

import pytest

from llama_verdict import config, derive, spec
from llama_verdict.backend import HttpBackend
from llama_verdict.decide import Decider

URL = config.backend_url()
MODEL = config.get("VERDICT_HEAD_MODEL") or "jev-omni:Q8_0"
LAYOUT = "jev-omni"

YES_NO = {"type": "noul"}
REFERENCE = [
    ("The meeting starts at 10 AM. It is now 9 AM.", "Has the meeting started?", YES_NO,
     {"true": 0.00011235327838221565, "false": 0.9998875856399536}),
    ("Customer: I was charged twice.\nAgent: I've refunded $29 to your card.\n"
     "Customer: Got it. All sorted, thanks!", "Was the issue actually resolved?", YES_NO,
     {"true": 0.9399133324623108, "false": 0.06008664518594742}),
    ("A fair six-sided die is rolled once.", "Which number comes up?",
     {"type": "choice", "criteria": dict.fromkeys("123456")},
     {"1": 0.23407739400863647, "2": 0.09292787313461304, "3": 0.04372488334774971,
      "4": 0.08901944011449814, "5": 0.08105313032865524, "6": 0.4591972529888153}),
    ("An urn holds one blue, one yellow, one red and one green marble. "
     "One is drawn without looking.", "Which marble is drawn?",
     {"type": "choice", "criteria": dict.fromkeys(["Blue", "Yellow", "Red", "Green"])},
     {"Blue": 0.41296717524528503, "Yellow": 0.12619410455226898,
      "Red": 0.0792793333530426, "Green": 0.38155943155288696}),
]


@pytest.fixture(scope="module")
def decider(tmp_path_factory):
    if not URL:
        pytest.skip("set LLAMA_VERDICT_URL")
    backend = HttpBackend(URL, MODEL)
    try:
        backend.props()
    except (OSError, http.client.HTTPException) as e:
        pytest.skip(f"{MODEL} is not served by the backend: {e}")
    # no layout named: the registry's readout says which head
    formatter, _ = derive.build(backend, MODEL,
                                cache_dir=str(tmp_path_factory.mktemp("formatters")))
    assert formatter.layout == LAYOUT
    return Decider(backend, formatter)


def test_the_head_answers_its_verification_at_every_option_count(decider):
    assert decider.formatter.verification["correct_ratio"] == 1.0


@pytest.mark.parametrize("state,question,body,reference", REFERENCE,
                         ids=["meeting", "refund", "die", "marbles"])
def test_the_gguf_lands_on_the_authors_published_probabilities(
        decider, state, question, body, reference):
    answer = decider.decide(state, {"q": dict(body, instructions=question)})["answers"]["q"]
    assert answer["probs"] == pytest.approx(
        reference, abs=spec.constants()["numeric_tolerance"]["probability_abs"])
    assert answer["option_mass"] is None and "no_option_mass" in answer["flags"]
