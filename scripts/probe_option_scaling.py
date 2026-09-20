#!/usr/bin/env python3
"""does the readout survive a long option list?

the browser element-table use case turns a page into dozens of options. this
measures whether option mass and accuracy hold as the list grows, and whether
the answer's position in the list biases the result.

usage:
  ./scripts/probe_option_scaling.py --base-url http://host:port \
      --models qwen3.5-0.8b:Q8_0 gemma-4-e2b-it:Q8_0
"""

import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import probe_prompt_layout as P
from probe_label_alphabet import opening_for

# a-z after A-Z, which the alphabet probe showed are all single tokens and all
# survive the assistant opening on both tokenizers tested
LABELS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]

STATE = (
    "Page: Flight search results for Lisbon to Reykjavik.\n"
    "The user wants to change when they are flying."
)
QUESTION = "Which element should be clicked to open the departure date picker?"
CORRECT = "Departure date field, currently showing 12 Mar"

DISTRACTORS = [
    "Site logo, links to the home page", "Primary navigation: Flights",
    "Primary navigation: Hotels", "Primary navigation: Car hire",
    "Account menu button", "Currency selector, showing EUR",
    "Language selector, showing English", "Search submit button",
    "Origin airport field, showing Lisbon", "Destination airport field, showing Reykjavik",
    "Return date field, currently showing 19 Mar", "Passenger count stepper",
    "Cabin class dropdown, showing Economy", "Direct flights only checkbox",
    "Sort control, showing Cheapest first", "Filter: maximum price slider",
    "Filter: departure time range", "Filter: airline checkbox list",
    "Filter: number of stops", "Result card 1, TAP Air Portugal, 08:15",
    "Result card 2, Icelandair, 11:40", "Result card 3, SAS, 14:05",
    "Result card 4, Lufthansa, 16:30", "Result card 5, KLM, 19:55",
    "Select fare button on result card 1", "Select fare button on result card 2",
    "Baggage information link", "Price alert toggle",
    "Share this search button", "Print results link",
    "Pagination: next page of results", "Pagination: previous page",
    "Footer link: Privacy policy", "Footer link: Terms of service",
    "Footer link: Contact support", "Footer link: Careers",
    "Cookie consent accept button", "Cookie consent manage preferences",
    "Newsletter signup email field", "Newsletter signup submit",
    "Live chat launcher", "Back to top button",
    "Recently viewed flights carousel", "Nearby airports suggestion",
    "Flexible dates calendar link", "Price history graph toggle",
    "Seat map preview link", "Travel requirements notice",
    "Promotional banner: 10 percent off", "Advertisement slot",
    "Breadcrumb: Home", "Breadcrumb: Flights",
]

COUNTS = [2, 4, 8, 16, 26, 52]
POSITIONS = ("first", "quarter", "middle", "last")


def build_options(n, position):
    """n options with the correct one placed at a chosen index."""
    idx = {"first": 0, "quarter": n // 4, "middle": n // 2, "last": n - 1}[position]
    picked = DISTRACTORS[: n - 1]
    opts = [(f"d{i}", d) for i, d in enumerate(picked)]
    opts.insert(idx, ("correct", CORRECT))
    return opts, idx


def formatter(opening):
    """the affixes that go with a given opening."""
    family = "qwen3.5" if opening.startswith("<|im_start|>") else "gemma4"
    return dict(P.FAMILIES[family], assistant_open=opening)


def score_wide(cli, opening, options):
    """the layout probe's scoring path, widened past its eight label alphabet."""
    saved = P.LABELS
    try:
        P.LABELS = LABELS
        return P.score(cli, formatter(opening), STATE, QUESTION, options)
    finally:
        P.LABELS = saved


def run(base_url, model):
    cli = P.Client(base_url, model)
    opening = opening_for(model)
    print(f"===== {model}")
    rows = []
    for n in COUNTS:
        for position in POSITIONS:
            options, idx = build_options(n, position)
            r = score_wide(cli, opening, options)
            ok = r["top"] == "correct"
            rows.append({"n": n, "position": position, "index": idx, "ok": ok,
                         "option_mass": r["option_mass"],
                         "confidence": max(r["probs"].values()) if r["probs"] else 0.0,
                         "n_probs_used": r["n_probs_used"]})
            print(f"  n={n:3} pos={position:8} idx={idx:3} "
                  f"{'ok ' if ok else 'MISS'} mass={r['option_mass']:.4f} "
                  f"conf={rows[-1]['confidence']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/option_scaling.json")
    args = ap.parse_args()

    report = {}
    for m in args.models:
        rows = run(args.base_url, m)
        report[m] = rows
        by_n = {}
        for r in rows:
            by_n.setdefault(r["n"], []).append(r)
        print(f"  --- {m} summary")
        for n, rs in sorted(by_n.items()):
            acc = sum(r["ok"] for r in rs)
            print(f"    n={n:3} correct {acc}/{len(rs)} "
                  f"min_mass={min(r['option_mass'] for r in rs):.4f} "
                  f"mean_conf={sum(r['confidence'] for r in rs) / len(rs):.3f}")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
