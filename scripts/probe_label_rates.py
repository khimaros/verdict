#!/usr/bin/env python3
"""how likely is the model to emit each label character at all?

docs/EVALS.md 2a measured that unfamiliar labels cost accuracy and option mass,
and that filtering them for visual distinctness made it worse because the
lookalikes are the COMMON characters. that names frequency as the mechanism
without measuring it directly.

this measures it directly. the whole-vocabulary readout verdict already makes
during derivation carries a probability per token, at exactly the position a
label is read from, and that number is currently discarded. it is the base rate
of each candidate label.

prints the base rate of three groups, which the probe scored separately:

  ascii     A-Za-z, 16/16 at option mass 0.9996
  wide      the first 52 non-ascii labels, 16/16 at 0.7780
  distinct  the first 52 with lookalikes dropped, 11/16 at 0.3146

if frequency is the mechanism these three separate cleanly, and the gap says
where a usability floor belongs. if they overlap, the explanation in 2a is
wrong and the floor has nothing to stand on.

MEASURED, AND THE FLOOR HAS NOTHING TO STAND ON. ascii comes out about 300x
more emittable than either alternative, which explains why the pinned 52
belong first. but `distinct` measures HIGHER than `wide` and scores far worse,
so base rate does not order the two non-ascii groups and a floor on it would
not have caught the one that failed. kept as the instrument that refuted it.

usage:
  scripts/probe_label_rates.py --model qwen3.5-9b:Q8_0
"""

import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

from llama_verdict import derive, spec
from llama_verdict.backend import HttpBackend
from probe_wide_labels import confusable
from smoke_models import OVERRIDES

GROUP = 52


def rates(backend, formatter):
    """base rate per single-character label, read where a label is scored."""
    opening = formatter.assistant_open
    out, _ = backend._post("/completion", {
        "prompt": opening, "n_predict": 1,
        "n_probs": spec.constants()["n_probs_ladder"][-1],
        "cache_prompt": True, "post_sampling_probs": False, "temperature": -1.0})
    entries = out["completion_probabilities"][0]["top_logprobs"]
    by_text = {}
    for e in entries:
        if len(e["token"]) == 1 and e["token"].isalpha():
            # several ids can detokenise to one character; the label carries
            # whichever id the alphabet picked, so keep the largest
            by_text[e["token"]] = max(by_text.get(e["token"], 0.0),
                                      math.exp(e["logprob"]))
    return by_text


def summarise(name, labels, by_text):
    present = [by_text.get(ell, 0.0) for ell in labels]
    nonzero = [p for p in present if p > 0]
    return {
        "group": name, "labels": len(labels),
        "absent_from_readout": sum(1 for p in present if p == 0),
        "median": statistics.median(present) if present else 0.0,
        "min": min(present) if present else 0.0,
        "max": max(present) if present else 0.0,
        "geometric_mean": (math.exp(statistics.fmean(math.log(p) for p in nonzero))
                           if nonzero else 0.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://10.1.200.250:7860")
    ap.add_argument("--model", default="qwen3.5-9b:Q8_0")
    ap.add_argument("--out", default="eval/results/label-rates.json")
    args = ap.parse_args()

    backend = HttpBackend(args.base_url, args.model)
    formatter, _ = derive.build(backend, args.model,
                                assistant_open=OVERRIDES.get(args.model))
    wide = derive.ensure_wide_labels(backend, formatter, 600)
    by_text = rates(backend, formatter)

    pinned = list(spec.constants()["labels"])
    non_ascii = [ell for ell in wide if not ell.isascii()]
    groups = {
        "ascii": pinned,
        "wide": non_ascii[:GROUP],
        "distinct": [e for e in non_ascii if not confusable(e)][:GROUP],
    }
    report = {"model": args.model,
              "single_char_letters_in_readout": len(by_text),
              "groups": [summarise(n, ells, by_text) for n, ells in groups.items()]}

    print(f"{args.model}: {len(by_text)} single-character letters in the readout\n")
    print(f"  {'group':<10} {'n':>4} {'absent':>7} {'min':>11} {'median':>11} "
          f"{'geo mean':>11}")
    for g in report["groups"]:
        print(f"  {g['group']:<10} {g['labels']:>4} {g['absent_from_readout']:>7} "
              f"{g['min']:>11.3e} {g['median']:>11.3e} {g['geometric_mean']:>11.3e}")

    out = os.path.join(HERE, "..", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
