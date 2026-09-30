#!/usr/bin/env python3
"""option mass across a set of models, derived at runtime for each.

the question this answers: is "can this model be scored by reading label
probabilities" a property that a public leaderboard already predicts? so far
the only model to fail was thinly measured, which is consistent with a weaker
claim -- that unmeasured models are unpredictable -- rather than with option
mass being orthogonal to quality. surveying models across the evidence range
separates the two.

each model costs one /props, ~104 tokenize calls and one /completion, plus
whatever llama-swap spends loading it.

usage:
  scripts/survey_option_mass.py --base-url http://host:7860 \
      --models qwen3.6-27b:Q8_0 gpt-oss-20b:Q8_0 --out eval/results/mass.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "python"))
from llama_verdict import config, derive
from llama_verdict.backend import HttpBackend


def survey(base_url, models, wide=0, layout=None):
    for model in models:
        started = time.monotonic()
        row = {"model": model}
        backend = HttpBackend(base_url, model)
        try:
            formatter, cached = derive.build(backend, model, layout=layout)
            row.update(option_mass=formatter.verification["mean_option_mass"],
                       correct=formatter.verification["smoke_correct"],
                       assistant_open=formatter.assistant_open,
                       template_sha256=formatter.template_sha256[:16],
                       cached=cached, usable=True)
            if wide:
                # the wide alphabet is a SEPARATE usability question and a
                # model can pass the first and fail this: the pinned labels
                # steer it perfectly while characters out of its own
                # vocabulary put the mass somewhere else. spec 10.1.
                ids = derive.ensure_wide_labels(backend, formatter, wide)
                row["wide"] = derive.verify_wide(backend, formatter, ids)
        except Exception as e:
            # a refusal is a result, not an error: it is the measurement
            row.update(usable=False, error=f"{type(e).__name__}: {e}")
        row["wall_s"] = round(time.monotonic() - started, 1)
        note = ""
        if row.get("wide"):
            w = row["wide"]
            note = (f"   wide {w['min_option_mass']:.4f} at "
                    f"{w['option_counts']} labels, {w['smoke_correct']}")
        print(f"{model:<32} "
              + (f"mass {row['option_mass']:.4f}  {row['correct']}"
                 if row.get("usable") else f"REFUSED  {row['error'][:70]}")
              + note + f"   ({row['wall_s']}s)", flush=True)
        yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    config.backend_url_argument(ap)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--wide", type=int, default=0, metavar="N",
                    help="also resolve N labels from the model's own "
                         "vocabulary and measure option mass on a list "
                         "labelled entirely from them (spec 10.1)")
    ap.add_argument("--layout", default=None,
                    help="one of spec/layouts, for every model, instead of "
                         "the readout the registry advertises")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = list(survey(args.base_url, args.models, args.wide, args.layout))
    failed = [r for r in rows if not r.get("usable")]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} cleared the floor", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
