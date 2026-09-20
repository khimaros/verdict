#!/usr/bin/env python3
"""drive browser-use/jev-ultrafast against a local verdict jev endpoint.

the decisions are made by a gguf model on your own llama-server. nothing about
the page or the goal is sent to a decision provider.

jev-ultrafast on main hardcodes the typesafe url, so this redirects it by
monkeypatching the one function that posts. that is deliberate: patching the
checkout would make the demo depend on an edited copy of somebody else's repo.

usage:
  ./demos/run_hn_demo.py --verdict http://127.0.0.1:8477 \
      --jev-ultrafast /tmp/jev-ultrafast
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gating import gate
from progress import Progress
from target_first import decide as decide_target_first
from shortlist import restore_answers, shortlist_request, shortlist_state

# worded in the operations that actually exist. jev-ultrafast's snapshot emits
# only scroll_down, scroll_up and wait as controls, plus CLICK and TYPE_TEXT on
# elements; there is no back, no new tab and no history navigation. asking to
# "go back to the front page" is asking for an operation the action space does
# not contain, and the model correctly answered BLOCKED. the way home is the
# Hacker News logo, so the goal has to say so.
DEFAULT_GOALS = [
    "Read the first 5 comments on each of the top 3 Hacker News stories.",
    "From the front page, click the comments link of the next story you have "
    "not visited yet.",
    "On a comments page, scroll down until you have seen the first 5 comments, "
    "then click the 'Hacker News' logo at the top left to return to the front "
    "page.",
    "You are done once you have read comments on three different stories.",
]

# both browser-use and browser-harness ship posthog telemetry that is on by
# default. a decision made locally that phones home is not made locally.
TELEMETRY_OFF = {
    "ANONYMIZED_TELEMETRY": "false",
    "BROWSER_HARNESS_TELEMETRY": "false",
    "BH_TELEMETRY": "false",
}


def redirect_to_verdict(base_url, timeout, dump_dir=None, keep=0, progress=None,
                        min_confidence=0.0, target_first=False):
    """send every decision to the local endpoint instead of api.typesafe.ai."""
    import httpx
    import jev_ultrafast.model as model

    original = model.post_json
    target = base_url.rstrip("/") + "/v1/systemone"

    # the upstream client fixes a 25s budget against a hosted service. a local
    # model on a one-slot server is slower than that, and a demo that dies on
    # the timeout tells you nothing about whether the decisions were any good.
    # this does not make the latency acceptable, it makes it measurable.
    model.CLIENT = httpx.Client(http2=True, timeout=timeout)

    stats = {"decisions": 0, "gated": 0, "decision_ms": []}
    # the proxy never sees the result of an action, only the next request. that
    # is enough: if the page is identical to the one the last decision acted
    # on, the action did nothing, and the operation that produced it is retired
    # for that page. the android agent learned this after eleven consecutive
    # identical scrolls at 0.81 to 0.96 confidence.
    memory = {"dead": {}, "acted": False, "last_page": None, "last_sig": None,
              "last_operation": None}

    def post_json(url, key, body):
        stats["decisions"] += 1
        counter = stats["decisions"]
        dropped = {}
        if progress:
            body, seen = progress.inject(body)
            if seen:
                print(f"  progress: {seen['stories_visited']}/{seen['stories_required']} "
                      f"visited {seen['already_visited']}", file=sys.stderr)
        if keep:
            body, reports, dropped = shortlist_request(body, keep)
            for name, r in reports.items():
                print(f"  shortlist {name}: {r['from']} -> {r['to']} "
                      f"({r['matched_goal']} matched the goal)", file=sys.stderr)
            # elements no question can choose are prefill nobody can act on
            body, trimmed = shortlist_state(body)
            if trimmed:
                print(f"  state elements: {trimmed['from']} -> {trimmed['to']}",
                      file=sys.stderr)
        if dump_dir:
            path = os.path.join(dump_dir, f"request-{counter:03d}.json")
            with open(path, "w") as f:
                json.dump(body, f, indent=2)
        page = body.get("state", {}).get("page", {}).get("url", "")
        sig = tuple(sorted(str(e.get("label", ""))
                           for e in body.get("state", {}).get("elements", [])))
        if (memory["last_operation"] and page == memory["last_page"]
                and sig == memory["last_sig"]):
            memory["dead"].setdefault(page, set()).add(memory["last_operation"])
            print(f"  page unchanged; {memory['last_operation']} retired here "
                  f"(dead: {sorted(memory['dead'][page])})", file=sys.stderr)

        started = time.monotonic()
        if target_first:
            # the same starvation the android agent had: the operation head
            # cannot see that a target head is certain unless it is told
            withhold = set(memory["dead"].get(page, ()))
            if not memory["acted"]:
                # a page nobody has acted on cannot already be finished
                withhold.add("DONE")
            answers = decide_target_first(
                body, lambda b: original(target, key, b)["answers"], withhold)
            result = {"model": "verdict", "answers": answers, "usage": {}}
        else:
            result = original(target, key, body)
        elapsed = (time.monotonic() - started) * 1000
        stats["decision_ms"].append(round(elapsed))
        # the client validates the distribution against the full set it sent
        result = restore_answers(result, dropped)
        if min_confidence:
            before = result["answers"].get("operation", {}).get("choice")
            result = gate(result, min_confidence)
            if result["answers"].get("operation", {}).get("choice") != before:
                stats["gated"] += 1
        chosen = result["answers"].get("operation", {}).get("choice")
        memory.update(last_page=page, last_sig=sig, last_operation=chosen)
        if chosen not in (None, "DONE", "BLOCKED"):
            memory["acted"] = True

        sizes = {name: len(q.get("criteria", {}))
                 for name, q in body["questions"].items()}
        print(f"  decision {counter:>3}  {elapsed:>7.0f} ms  options={sizes}",
              file=sys.stderr)
        return result

    model.post_json = post_json
    return target, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict", default="http://127.0.0.1:8477")
    ap.add_argument("--url", default="https://news.ycombinator.com/")
    ap.add_argument("--goal", action="append", default=None)
    ap.add_argument("--cdp", default=os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222"))
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="http budget per decision; upstream fixes 25s against a hosted service")
    ap.add_argument("--dump-dir", default=None, help="write each request body here")
    ap.add_argument("--target-first", action="store_true",
                    help="ask which target first, then ask which operation "
                         "knowing what each would hit. on android this moved "
                         "the correct operation from 0.13 to 0.9955")
    ap.add_argument("--min-confidence", type=float, default=0.0, metavar="P",
                    help="stop rather than act when the chosen target is below P. "
                         "the scorer already reports this; nothing else reads it")
    ap.add_argument("--track-progress", type=int, default=0, metavar="N",
                    help="carry a visited-story count in the state. the harness "
                         "computes it; no model is involved")
    ap.add_argument("--shortlist", type=int, default=0, metavar="N",
                    help="cut oversized candidate lists to N before deciding. "
                         "changes the answer set: a removed option cannot be chosen")
    ap.add_argument("--summary", default=None,
                    help="write a machine readable outcome here, for repeat runs")
    args = ap.parse_args()

    os.environ.update(TELEMETRY_OFF)
    os.environ["BU_CDP_URL"] = args.cdp
    # the endpoint does not require a key, but the client insists on sending one
    os.environ.setdefault("TYPESAFE_API_KEY", "local")
    os.environ.setdefault("TYPESAFE_MODEL", "jev-latest")

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    required = args.track_progress or 3
    tracker = Progress(args.track_progress) if args.track_progress else None
    target, stats = redirect_to_verdict(args.verdict, args.timeout, args.dump_dir,
                                        args.shortlist, tracker, args.min_confidence,
                                        target_first=args.target_first)
    print(f"decisions -> {target}", file=sys.stderr)
    print(f"chrome    -> {args.cdp}", file=sys.stderr)

    from jev_ultrafast import Agent

    goals = args.goal or DEFAULT_GOALS
    outcome = {"status": "error", "actions": 0, "stories_visited": 0,
               "stories_required": required, "final_url": None, "error": None}
    state = None
    try:
        with Agent(args.url, goals) as agent:
            for state in agent.run():
                history = state["history"]
                last = history[-1] if history else {}
                print(f"{state['elapsed_ms']:>7} ms  {len(history):>3} actions  "
                      f"{state['status']:<12} {last.get('action', '')}", file=sys.stderr)
    except Exception as e:
        # a crashed run is an outcome, not a gap in the sample
        outcome["error"] = f"{type(e).__name__}: {e}"
        print(f"run failed: {outcome['error']}", file=sys.stderr)

    if state:
        outcome.update(status=state["status"], actions=len(state["history"]),
                       final_url=state["page"]["url"],
                       elapsed_ms=state.get("elapsed_ms"))
        print(f"\nfinal url: {state['page']['url']}", file=sys.stderr)
    if tracker:
        # ground truth for the task: distinct story pages actually reached
        outcome["stories_visited"] = len(tracker.visited)
        outcome["stories"] = [v["title"] for v in tracker.visited]
    outcome.update(decisions=stats["decisions"], gated=stats["gated"],
                   decision_ms=stats["decision_ms"])

    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(outcome, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
