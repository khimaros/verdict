#!/usr/bin/env python3
"""can this model decide at all? one cheap pass per model, before any agent.

two live-site sweeps were spent learning things this answers in seconds. an
agent run against hacker news measures the model, the harness, the network and
a page that changes underneath it, and when the number is bad it does not say
which. `scripts/eval_interventions.py` already had the right instrument --
element selection with ground truth, fixed distractors, the correct option
placed at four positions per count -- and it needs no browser and no device.

so this is the gate an agent sweep should sit behind:

  derives  the formatter comes from the model's own template
  mass     the readout is trustworthy at all
  accuracy 16 cases, 4 option counts x 4 positions
  bias     accuracy by position, since a model that only picks `first` scores
           well on a benchmark whose answer is often first

usage:
  scripts/smoke_models.py --models qwen3.5-9b:Q8_0 minicpm5-2b:Q8_0
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

from eval_interventions import COUNTS, POSITIONS, STATE, case
from llama_verdict import derive
from llama_verdict.backend import HttpBackend
from llama_verdict.decide import Decider

OVERRIDES = {"gpt-oss-20b:Q8_0": "<|start|>assistant<|channel|>final<|message|>",
             "gpt-oss-120b:Q8_0": "<|start|>assistant<|channel|>final<|message|>"}


def smoke(base_url, model, counts, order_averaging=1, wide_alphabet=False):
    """`order_averaging` scores each option list forward and reversed and
    averages. a bias tied to WHERE an option sits cancels, because the correct
    answer sits at index p in one order and n-1-p in the other. it costs one
    extra pass per order and is the only mechanism here aimed at position.

    `wide_alphabet` labels lists past 52 from the model's own vocabulary and
    reads them in one exact pass. it is the only way to measure a long list
    without the tournament's partitioning in the way, and the number it
    produces is what says whether an unfamiliar label is a usable label.
    """
    backend = HttpBackend(base_url, model)
    started = time.monotonic()
    formatter, _ = derive.build(backend, model,
                                assistant_open=OVERRIDES.get(model))
    decider = Decider(backend, formatter, tournament=True,
                      order_averaging=order_averaging,
                      wide_alphabet=wide_alphabet)

    rows = []
    for n in counts:
        for position in POSITIONS:
            t0 = time.monotonic()
            answer = decider.decide(STATE, case(n, position))["answers"]["pick"]
            rows.append({"n": n, "position": position,
                         "ok": answer["top"] == "correct",
                         "correct_prob": answer["probs"].get("correct", 0.0),
                         "option_mass": answer["option_mass"],
                         "flags": answer["flags"],
                         "ms": (time.monotonic() - t0) * 1000.0})

    ok = sum(r["ok"] for r in rows)
    return {
        "model": model,
        "order_averaging": order_averaging,
        "wide_alphabet": wide_alphabet,
        "correct": f"{ok}/{len(rows)}",
        "accuracy": ok / len(rows),
        "by_position": {p: sum(r["ok"] for r in rows if r["position"] == p)
                        for p in POSITIONS},
        "by_count": {n: sum(r["ok"] for r in rows if r["n"] == n) for n in counts},
        "min_option_mass": min(r["option_mass"] for r in rows),
        "median_ms": statistics.median(r["ms"] for r in rows),
        "wall_s": round(time.monotonic() - started, 1),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://10.1.200.250:7860")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--counts", type=int, nargs="+", default=COUNTS)
    ap.add_argument("--order-averaging", type=int, default=1,
                    help="score under N option orders and average. 2 is "
                         "forward and reversed, which is the cheapest pair "
                         "that disagrees about position")
    ap.add_argument("--wide-alphabet", action="store_true",
                    help="label lists past 52 from the model's own vocabulary "
                         "and read them in one exact pass")
    ap.add_argument("--out", default="eval/results/smoke-models.json")
    args = ap.parse_args()

    report = []
    for model in args.models:
        try:
            row = smoke(args.base_url, model, args.counts,
                        args.order_averaging, args.wide_alphabet)
            per_n = " ".join(f"{n}:{c}" for n, c in row["by_count"].items())
            per_p = " ".join(f"{p[:3]}:{c}" for p, c in row["by_position"].items())
            print(f"{model:<30} {row['correct']:>6}  mass>={row['min_option_mass']:.4f}  "
                  f"{row['median_ms']:>6.0f}ms  [{per_n}]  [{per_p}]", flush=True)
        except Exception as e:
            row = {"model": model, "error": f"{type(e).__name__}: {e}"}
            print(f"{model:<30} REFUSED  {row['error'][:70]}", flush=True)
        report.append(row)

    os.makedirs(os.path.dirname(os.path.join(HERE, "..", args.out)), exist_ok=True)
    with open(os.path.join(HERE, "..", args.out), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
