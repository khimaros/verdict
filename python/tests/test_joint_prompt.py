"""the joint schema prompt, against the author's own encoder.

a joint head scores each option by where it SITS in the prompt, so the prompt
builder has to return token spans as well as tokens, and one token out is a
different option. the author tokenizes every piece separately and joins the
ids, which is what makes the spans exact; verdict does the same.

`scripts/make_joint_head_fixture.py` ran `encode_record` from
Cloudflare/clef-flash @ 17f0b0a over these records with a tokenizer of one
token per character, so what is pinned is the pieces and their order.
"""

import json
import os

import pytest

from llama_verdict import joint_prompt

LAYOUT = "clef-flash"
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "joint_prompt.json")

with open(FIXTURE) as f:
    CASES = json.load(f)["cases"]


def characters(text):
    return [ord(c) for c in text]


def encode(case):
    record = case["record"]
    return joint_prompt.encode(characters, LAYOUT, record["state"], record["questions"],
                               max_length=case.get("max_length"))


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_the_tokens_are_the_authors(case):
    tokens, _ = encode(case)
    assert "".join(map(chr, tokens)) == "".join(map(chr, case["input_ids"]))


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_question_and_option_sits_where_the_author_put_it(case):
    _, fields = encode(case)
    assert [{"id": f.name, "type": f.wire_type, "question_span": list(f.spans.question),
             "option_spans": [list(s) for s in f.spans.options],
             "option_ids": list(f.option_ids)} for f in fields] == case["questions"]


def test_a_span_holds_exactly_the_option_the_head_scores():
    tokens, fields = encode(CASES[0])
    queue = next(f for f in fields if f.name == "queue")
    start, end = queue.spans.options[1]
    assert "".join(map(chr, tokens[start:end])) == (
        '{"description":"Invoices.","option_id":"billing"}')
    assert queue.spans.kind == 1


def test_a_schema_too_long_for_the_window_is_refused():
    record = CASES[0]["record"]
    with pytest.raises(ValueError, match="state"):
        joint_prompt.encode(characters, LAYOUT, record["state"], record["questions"],
                            max_length=100)
