#!/usr/bin/env python3
"""does a debiasing intervention actually help?

element selection with ground truth: one correct target among plausible page
distractors, placed at four positions per option count, so position bias shows
up as a pattern rather than as noise.

this is the benchmark the browser demo cannot be, because a live site changes
underneath every run.

usage:
  ./scripts/eval_interventions.py --base-url http://host:port \
      --model qwen3.5-9b:Q8_0 --formatter spec/formatters/qwen3.5-9b_Q8_0.json
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from llama_verdict.backend import HttpBackend
from llama_verdict.decide import Decider
from llama_verdict.types import Formatter

STATE = ("Page: Flight search results for Lisbon to Reykjavik.\n"
         "The user wants to change when they are flying.")
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
    # past 52 the wide alphabet is under test and the list has to stay a real
    # page. filler of the form "element 57" would leave the target as the only
    # plausible entry and the long-list score would measure nothing.
    "Result card 6, Air France, 06:20", "Result card 7, British Airways, 09:10",
    "Result card 8, Iberia, 12:45", "Result card 9, Norwegian, 15:30",
    "Result card 10, Finnair, 18:05", "Result card 11, Ryanair, 20:40",
    "Result card 12, easyJet, 22:15", "Select fare button on result card 3",
    "Select fare button on result card 4", "Select fare button on result card 5",
    "Select fare button on result card 6", "Select fare button on result card 7",
    "Fare details expander on result card 1", "Fare details expander on result card 2",
    "Fare details expander on result card 3", "Emissions estimate on result card 1",
    "Emissions estimate on result card 2", "Layover detail on result card 4",
    "Aircraft type label on result card 5", "Operated-by notice on result card 6",
    "Filter: departure airport list", "Filter: arrival airport list",
    "Filter: layover duration range", "Filter: layover airport list",
    "Filter: aircraft type list", "Filter: fare type, Economy Basic",
    "Filter: fare type, Economy Flex", "Filter: baggage included checkbox",
    "Filter: refundable fares only", "Filter: exclude overnight layovers",
    "Filter: arrival time range", "Filter: reset all filters",
    "Sort control, Fastest first", "Sort control, Earliest departure",
    "Sort control, Best value", "View as calendar toggle",
    "View as grid toggle", "Compare selected fares button",
    "Save this search button", "Email these results link",
    "Currency selector, showing USD", "Currency selector, showing GBP",
    "Language selector, showing Portuguese", "Accessibility statement link",
    "Skip to main content link", "Open the help centre",
    "Manage my booking link", "Check in online link",
    "Flight status lookup link", "Airport guide link",
    "Travel insurance offer", "Car hire cross-sell panel",
    "Hotel cross-sell panel", "Airport transfer cross-sell panel",
    "Loyalty programme signup", "Student discount notice",
    "Group booking enquiry link", "Corporate travel link",
    "Gift card purchase link", "Mobile app download banner",
    "Social link: follow on X", "Social link: follow on Facebook",
    "Footer link: About us", "Footer link: Press",
    "Footer link: Investor relations", "Footer link: Sitemap",
    "Footer link: Cookie settings", "Footer link: Modern slavery statement",
    "Region selector, showing Europe", "Session timeout notice",
    "Search history: Lisbon to Oslo", "Search history: Porto to Reykjavik",
    "Nearby date suggestion, 11 Mar", "Nearby date suggestion, 13 Mar",
    "Fare calendar link for March", "Cheapest month suggestion",
]

COUNTS = [8, 16, 26, 52]
POSITIONS = ("first", "quarter", "middle", "last")

INTERVENTIONS = {
    "baseline": {},
    "order-averaged": {"order_averaging": 2},
    "prior-corrected": {"prior_correction": True},
    "both": {"order_averaging": 2, "prior_correction": True},
}


def case(n, position):
    """n options with the correct one at a chosen index, page order preserved."""
    index = {"first": 0, "quarter": n // 4, "middle": n // 2, "last": n - 1}[position]
    ids = [f"d{i:02d}" for i in range(n - 1)]
    criteria = dict(zip(ids, DISTRACTORS[: n - 1], strict=True))
    items = list(criteria.items())
    items.insert(index, ("correct", CORRECT))
    return {"pick": {"type": "choice", "instructions": QUESTION, "criteria": dict(items)}}


def run(decider, cases):
    rows = []
    for n, position in cases:
        questions = case(n, position)
        t0 = time.monotonic()
        answer = decider.decide(STATE, questions)["answers"]["pick"]
        rows.append({
            "n": n, "position": position,
            "ok": answer["top"] == "correct",
            "confidence": answer["confidence"],
            "correct_prob": answer["probs"].get("correct", 0.0),
            "option_mass": answer["option_mass"],
            "ms": (time.monotonic() - t0) * 1000.0,
        })
    return rows


def summarise(name, rows):
    ok = sum(r["ok"] for r in rows)
    by_position = {}
    for p in POSITIONS:
        subset = [r for r in rows if r["position"] == p]
        by_position[p] = f"{sum(r['ok'] for r in subset)}/{len(subset)}"
    return {
        "intervention": name,
        "correct": f"{ok}/{len(rows)}",
        "accuracy": ok / len(rows),
        "by_position": by_position,
        "mean_correct_prob": statistics.mean(r["correct_prob"] for r in rows),
        "min_option_mass": min(r["option_mass"] for r in rows),
        "mean_ms": statistics.mean(r["ms"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--formatter", required=True)
    ap.add_argument("--counts", type=int, nargs="+", default=COUNTS)
    ap.add_argument("--only", nargs="+", default=list(INTERVENTIONS))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "eval", "results"))
    args = ap.parse_args()

    formatter = Formatter.load(args.formatter)
    cases = [(n, p) for n in args.counts for p in POSITIONS]
    print(f"{len(cases)} cases per intervention, counts {args.counts}\n")

    summaries, raw = [], {}
    for name in args.only:
        kwargs = INTERVENTIONS[name]
        decider = Decider(HttpBackend(args.base_url, args.model), formatter, **kwargs)
        rows = run(decider, cases)
        raw[name] = rows
        s = summarise(name, rows)
        summaries.append(s)
        print(f"  {name:<16} {s['correct']:>7}  "
              f"P(correct)={s['mean_correct_prob']:.3f}  "
              f"mass>={s['min_option_mass']:.4f}  {s['mean_ms']:>6.0f} ms  "
              f"by position {s['by_position']}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"interventions-{args.model.replace(':', '_')}.json")
    with open(path, "w") as f:
        json.dump({"model": args.model, "counts": args.counts,
                   "summaries": summaries, "rows": raw}, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
