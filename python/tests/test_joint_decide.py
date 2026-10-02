"""a decision through a joint head: one prompt, one read, every question.

a joint layout does not fit the one-suffix-per-question loop. all the
questions of a request go into one prompt, the backend hands back the hidden
state of every position once, and the head decides them together. these cases
run the real prompt builder and the real head arithmetic at toy size against a
backend that returns hidden states computed from the tokens it was sent.
"""

import dataclasses
import json
import os

import pytest

np = pytest.importorskip("numpy")

from llama_verdict import joint_head, joint_prompt, types  # noqa: E402
from llama_verdict.decide import Decider  # noqa: E402

LAYOUT = "clef-flash"
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "joint_head.json")
TICKET = "My invoice charged me twice for the same month and I want a refund."
QUESTIONS = {
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Not urgent.", "Soon.", "Now."]},
    "queue": {"type": "choice", "instructions": "Which queue?",
              "criteria": {"technical": "Bugs.", "billing": "Invoices.", "access": None}},
    "refund": {"type": "noul", "instructions": "Do they want money back?"},
}


def states(tokens, width):
    """a hidden state per position that depends on the token and where it is."""
    return np.sin(np.outer(np.asarray(tokens) + np.arange(len(tokens)),
                           np.arange(1, width + 1)) / 7.0).astype(np.float32)


class EveryPosition:
    """a backend serving every position's hidden state, as pooling none does."""
    model = "clef-flash:Q8_0"

    def __init__(self, width):
        self.width, self.read = width, []

    def tokenize(self, text, add_special=False):
        return [ord(c) for c in text]

    def hidden_states(self, tokens):
        self.read.append(list(tokens))
        return {"hidden": states(tokens, self.width).tolist(), "score_ms": 1.0}


@pytest.fixture(scope="module")
def toy():
    with open(FIXTURE) as f:
        golden = json.load(f)
    tensors = {name: np.asarray(t["values"], dtype=np.float32).reshape(t["shape"])
               for name, t in golden["tensors"].items()}
    embedding = np.asarray(golden["embedding"]["values"], dtype=np.float32).reshape(
        golden["embedding"]["shape"])
    return joint_head.JointSchema(tensors, golden["config"]["heads"],
                                  rows=lambda ids: embedding[np.asarray(ids) % len(embedding)])


def decider(toy):
    formatter = dataclasses.replace(
        types.Formatter.for_layout(LAYOUT, None),
        verification={"smoke_correct": "3/3", "correct_ratio": 1.0})
    backend = EveryPosition(toy.t["hidden_norm.weight"].shape[0])
    return Decider(backend, formatter, head=toy), backend


def test_every_question_is_answered_from_one_read_of_one_prompt(toy):
    d, backend = decider(toy)
    result = d.decide(TICKET, QUESTIONS)
    tokens, fields = joint_prompt.encode(lambda t: [ord(c) for c in t], LAYOUT, TICKET, QUESTIONS)
    assert backend.read == [tokens]
    assert result["usage"]["input_tokens"] == len(tokens)
    expected = toy.probabilities(toy.logits(states(tokens, backend.width), tokens,
                                            [f.spans for f in fields]))
    assert list(result["answers"]) == list(QUESTIONS)
    for field, row in zip(fields, expected, strict=True):
        answer = result["answers"][field.name]
        assert answer["probs"] == pytest.approx(dict(zip(field.option_ids, row, strict=True)),
                                                abs=1e-6)
        assert answer["option_mass"] is None and "no_option_mass" in answer["flags"]
        assert answer["provenance"]["readout"] == "joint-head"


def test_the_answers_keep_their_types(toy):
    d, _ = decider(toy)
    answers = d.decide(TICKET, QUESTIONS)["answers"]
    assert answers["urgency"]["type"] == "score"
    assert answers["urgency"]["legend"] == {"0": "Not urgent.", "1": "Soon.", "2": "Now."}
    assert answers["urgency"]["score"] == pytest.approx(
        sum(int(level) * p for level, p in answers["urgency"]["probs"].items()))
    assert set(answers["refund"]["probs"]) == {"true", "false"}
    assert answers["queue"]["top"] in QUESTIONS["queue"]["criteria"]
