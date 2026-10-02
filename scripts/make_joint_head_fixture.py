#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""write the golden fixture verdict's joint schema head is held to.

clef's decision head is a small transformer whose only published
implementation is the author's torch class. verdict ports it to numpy, and a
port that is slightly wrong still returns confident distributions, so the port
is tested against what the AUTHOR'S class computes: this script fetches
`joint_schema_model.py` at a pinned revision, builds `JointSchemaHead` at toy
size with seeded random weights, runs one record through it and records the
weights, the inputs and the logits. torch is needed here and nowhere else.

every parameter is drawn at random, including the three the author initialises
to zero, since a zero gate would hide the half of the arithmetic behind it.

usage:
  scripts/make_joint_head_fixture.py     # needs uv; rewrites the fixture
"""

import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from llama_verdict import head

OUT = os.path.join(HERE, "..", "python", "tests", "fixtures", "joint_head.json")
AUTHOR = {"repo": "Cloudflare/clef-flash",
          "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
          "file": "joint_schema_model.py",
          "sha256": "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"}
SEED = 20261001
CONFIG = {"hidden_size": 8, "width": 8, "routing_layers": 1, "layers": 2, "heads": 2,
          "feedforward": 16}
VOCAB, POSITIONS = 20, 14
# a choice of three and a boolean, as (type, question span, option spans)
QUESTIONS = [(1, (2, 4), ((4, 6), (6, 7), (7, 9))), (0, (9, 10), ((10, 12), (12, 13)))]
PARAMETER_SPREAD = 0.5
PROMPT_OUT = os.path.join(os.path.dirname(OUT), "joint_prompt.json")
TICKET = "My invoice charged me twice for the same month and I want a refund."
# records for the author's prompt builder: every question type, criteria that
# arrive unsorted, missing and structured instructions, a described and an
# undescribed option, a structured state, and a state too long for the window
PROMPT_CASES = [
    {"id": "every-type", "record": {"state": TICKET, "questions": {
        "refund": {"type": "noul", "instructions": "Do they want money back?"},
        "churn": {"type": "noul", "instructions": "Do they threaten to cancel?",
                  "criteria": {"true": "They say they will leave."}},
        "queue": {"type": "choice", "instructions": "Which queue should handle this?",
                  "criteria": {"technical": "Bugs and outages.", "billing": "Invoices.",
                               "access": None}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["Not urgent.", "Soon.", "Now."]}}}},
    {"id": "instructions-missing-and-structured", "record": {"state": TICKET, "questions": {
        "is_billing": {"type": "noul"},
        "empty": {"type": "noul", "instructions": ""},
        "ruled": {"type": "choice", "instructions": {"rules": ["pick one"], "goal": "route"},
                  "criteria": {"b": {"label": "second"}, "a": "first"}}}}},
    {"id": "state-object", "record": {
        "state": {"subject": "Stripe sync broken", "plan": "pro", "note": "caf" + chr(233)},
        "questions": {"outage": {"type": "noul", "instructions": "Is this an outage?"}}}},
    {"id": "state-truncated", "max_length": 800, "record": {
        "state": TICKET * 8,
        "questions": {"refund": {"type": "noul", "instructions": "Do they want money back?"}}}},
]

def authors_module():
    spec = importlib.util.spec_from_file_location("joint_schema_model", head.fetch(AUTHOR))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def flat(tensor):
    return {"shape": list(tensor.shape), "values": tensor.flatten().tolist()}


def main():
    author = authors_module()
    torch.manual_seed(SEED)
    model = author.JointSchemaHead(**CONFIG).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=PARAMETER_SPREAD)
    hidden = torch.randn(1, POSITIONS, CONFIG["hidden_size"])
    input_ids = torch.randint(0, VOCAB, (1, POSITIONS))
    embedding = torch.randn(VOCAB, CONFIG["hidden_size"])
    record = author.EncodedRecord(
        input_ids=tuple(input_ids[0].tolist()), record_id="fixture",
        questions=tuple(author.EncodedQuestion(
            question_id=f"q{i}", question_type=kind, question_span=span, option_spans=options,
            option_ids=tuple(str(n) for n in range(len(options))))
            for i, (kind, span, options) in enumerate(QUESTIONS)))
    with torch.inference_mode():
        logits = model(hidden, input_ids, torch.ones(1, POSITIONS, dtype=torch.long),
                       [record], embedding)[0]
    fixture = {
        "source": f"{AUTHOR['repo']} @ {AUTHOR['revision'][:7]}: {AUTHOR['file']} "
                  f"JointSchemaHead.forward()",
        "torch": torch.__version__, "seed": SEED, "config": CONFIG,
        "tensors": {name: flat(value) for name, value in model.state_dict().items()},
        "hidden": flat(hidden[0]), "input_ids": input_ids[0].tolist(),
        "embedding": flat(embedding),
        "questions": [{"type": kind, "question_span": list(span),
                       "option_spans": [list(o) for o in options]}
                      for kind, span, options in QUESTIONS],
        "logits": [q.tolist() for q in logits]}
    write(OUT, fixture)
    print(f"wrote {os.path.relpath(OUT)}: {len(fixture['tensors'])} tensors, "
          f"logits {fixture['logits']}")
    prompts = [encoded(author, case) for case in PROMPT_CASES]
    write(PROMPT_OUT, {"source": f"{AUTHOR['repo']} @ {AUTHOR['revision'][:7]}: "
                                 f"{AUTHOR['file']} encode_record()",
                       "tokenizer": "one token per character, its code point",
                       "cases": prompts})
    print(f"wrote {os.path.relpath(PROMPT_OUT)}: {len(prompts)} records")


class Characters:
    """a tokenizer of one token per character, so the fixture pins the pieces
    the author tokenizes and their order rather than any real vocabulary."""

    def __call__(self, text, add_special_tokens=False):
        return type("Encoding", (), {"input_ids": [ord(c) for c in text]})


def encoded(author, case):
    record = author.encode_record(Characters(), case["record"],
                                  max_length=case.get("max_length", 16384))
    types = {number: name for name, number in author.QUESTION_TYPES.items()}
    return {**case, "input_ids": list(record.input_ids),
            "questions": [{"id": q.question_id, "type": types[q.question_type],
                           "question_span": list(q.question_span),
                           "option_spans": [list(s) for s in q.option_spans],
                           "option_ids": list(q.option_ids)} for q in record.questions]}


def write(path, fixture):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(fixture, indent=1) + "\n")


if __name__ == "__main__":
    main()
