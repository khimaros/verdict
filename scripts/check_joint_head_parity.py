#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "safetensors"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""is verdict's joint head the author's, at full size and on real inputs?

`tests/test_joint_head.py` holds the numpy port to the author's torch class at
toy size with random weights. this runs both on the PUBLISHED weights and on
hidden states a live llama-server returns for real prompts, and reports how far
apart their logits and probabilities land.

it checks the head and nothing upstream of it: both arms read the same hidden
states, so whether the gguf's states are the bf16 torch backbone's is a
separate question this does not answer.

usage:
  scripts/check_joint_head_parity.py [--model clef-flash:Q8_0] [--layout clef-flash]
"""

import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from llama_verdict import config, head, joint_head, joint_prompt, spec
from llama_verdict.backend import HttpBackend
from make_joint_head_fixture import PROMPT_CASES, authors_module

# float32 against float32 over six attention layers; a wrong port is off by
# whole logits, not by this
TOLERANCE = 1e-3
OPTION_COUNTS = (26, 52)


def cases():
    """the prompt fixture's records, plus long option lists."""
    for case in PROMPT_CASES:
        yield case["id"], case["record"], case.get("max_length")
    for count in OPTION_COUNTS:
        criteria = {f"element_{i}": f"the element numbered {i}" for i in range(count)}
        yield f"choice-of-{count}", {
            "state": f"The user asked for the element numbered {count // 2}.",
            "questions": {"pick": {"type": "choice", "instructions": "Which element?",
                                   "criteria": criteria}}}, None


def authors_head(author, path, declared):
    """the author's class at the size the published weights imply."""
    weights = load_file(path)
    width, hidden = weights["memory_projection.weight"].shape

    def count(prefix):
        return len({k.split(".")[1] for k in weights if k.startswith(prefix + ".")})

    model = author.JointSchemaHead(
        hidden_size=hidden, width=width, routing_layers=count("evidence_layers"),
        layers=count("layers"), heads=declared["heads"],
        feedforward=weights["layers.0.linear1.weight"].shape[0])
    model.load_state_dict(weights, strict=True)
    return model.eval()


def authors_logits(author, model, hidden, tokens, fields, rows):
    """the author's forward. it reads the output embedding matrix only at the
    option tokens, so those rows stand in for the matrix under renumbered ids."""
    used = sorted({t for f in fields for s, e in f.spans.options for t in tokens[s:e]})
    renumbered = [used.index(t) if t in used else 0 for t in tokens]
    record = author.EncodedRecord(
        input_ids=tuple(renumbered), record_id="parity",
        questions=tuple(author.EncodedQuestion(
            question_id=f.name, question_type=f.spans.kind, question_span=f.spans.question,
            option_spans=tuple(f.spans.options), option_ids=tuple(f.option_ids))
            for f in fields))
    with torch.inference_mode():
        out = model(torch.from_numpy(hidden)[None], torch.tensor([renumbered]),
                    torch.ones(1, len(tokens), dtype=torch.long), [record],
                    torch.from_numpy(rows(used)))[0]
    return [q.numpy() for q in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default=config.backend_url())
    ap.add_argument("--model", default="clef-flash:Q8_0")
    ap.add_argument("--layout", default="clef-flash")
    args = ap.parse_args()
    declared = spec.load_layout(args.layout)["head"]
    path = head.fetch(declared)
    ours = joint_head.load(path, declared)
    author = authors_module()
    theirs = authors_head(author, path, declared)
    backend = HttpBackend(args.base_url, args.model)
    worst = {"logit": 0.0, "probability": 0.0}
    agree = total = 0
    for name, record, max_length in cases():
        tokens, fields = joint_prompt.encode(backend.tokenize, args.layout, record["state"],
                                             record["questions"], max_length)
        hidden = np.asarray(backend.hidden_states(tokens)["hidden"], dtype=np.float32)
        mine = ours.logits(hidden, tokens, [f.spans for f in fields])
        reference = authors_logits(author, theirs, hidden, tokens, fields, ours.rows)
        for a, b in zip(mine, reference, strict=True):
            gap = {"logit": float(np.abs(a - b).max()),
                   "probability": float(np.abs(joint_head.softmax(a)
                                               - joint_head.softmax(b)).max())}
            worst = {k: max(worst[k], gap[k]) for k in worst}
            agree += int(a.argmax() == b.argmax())
            total += 1
        print(f"{name}: {len(tokens)} tokens, {len(fields)} questions")
    print(f"{total} questions: same answer on {agree}, worst logit gap {worst['logit']:.2e}, "
          f"worst probability gap {worst['probability']:.2e}")
    return 0 if agree == total and worst["probability"] < TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
