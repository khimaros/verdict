#!/usr/bin/env python3
"""how many option labels can a model actually offer?

a label is usable only if it is a single token AND does not merge with the
assistant opening that precedes it, because the label is scored at exactly
that position. the ceiling is a tokenizer fact, so it has to be measured per
model rather than assumed.

usage:
  ./scripts/probe_label_alphabet.py --base-url http://host:port \
      --models qwen3.5-0.8b:Q8_0 gemma-4-e2b-it:Q8_0
"""

import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import probe_prompt_layout as P

# each model's own no-think generation prompt. see docs/DECISIONS.md phase 0b:
# scoring against another model's opening collapses option mass by ~7 orders.
OPENINGS = {
    "qwen3.5": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "qwen3.6": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "gemma-4-e2b": "<|turn>model\n",
    "gemma-4-e4b": "<|turn>model\n",
    "gemma-4-12b": "<|turn>model\n<|channel>thought\n<channel|>",
    "gemma-4-31b": "<|turn>model\n<|channel>thought\n<channel|>",
}

ALPHABETS = {
    "upper A-Z": [chr(65 + i) for i in range(26)],
    "lower a-z": [chr(97 + i) for i in range(26)],
    "digits 0-9": [str(i) for i in range(10)],
    "numbers 10-99": [str(i) for i in range(10, 100)],
}


def opening_for(model):
    match = max((p for p in OPENINGS if model.startswith(p)), key=len, default=None)
    if not match:
        raise SystemExit(f"no known generation prompt for {model}")
    return OPENINGS[match]


def usable(cli, opening, candidates):
    """single token, and still its own token after the assistant opening."""
    open_tok = cli.tokenize(opening)
    good, multi, merged = [], [], []
    for c in candidates:
        tok = cli.tokenize(c)
        if len(tok) != 1:
            multi.append(c)
        elif cli.tokenize(opening + c) != open_tok + tok:
            merged.append(c)
        else:
            good.append(c)
    return good, multi, merged


def run(base_url, model):
    cli = P.Client(base_url, model)
    opening = opening_for(model)
    print(f"===== {model}")
    print(f"  opening {opening!r}")
    total = 0
    out = {}
    for name, alpha in ALPHABETS.items():
        good, multi, merged = usable(cli, opening, alpha)
        total += len(good)
        out[name] = good
        note = ""
        if multi:
            note += f" multi-token={len(multi)}"
        if merged:
            note += f" merged={merged[:8]}"
        print(f"  {name:14} usable {len(good):3}/{len(alpha):3}{note}")
    print(f"  ---> {total} usable labels total")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/label_alphabet.json")
    args = ap.parse_args()

    report = {m: run(args.base_url, m) for m in args.models}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
