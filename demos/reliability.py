#!/usr/bin/env python3
"""run a browser demo several times and report how often it works.

one run of an agent against a live site is an anecdote. the front page changes,
the model is deterministic but the page is not, and a single success and a
single failure are equally uninformative.

reports the ground truth the task actually asks for: how many distinct story
pages were reached, out of the three required. for `browser_agent.py` that is
counted from what was READ off each page rather than from pages the agent
believes it opened -- a url it navigated to and collected nothing from did not
answer anything.

usage:
  ./demos/reliability.py --runs 5 -- --shortlist 26 --track-progress 3
  ./demos/reliability.py --demo browser_agent.py --runs 5 -- --goal ... --collect ...
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CRASHED = {"status": "crashed", "stories_visited": 0, "actions": 0,
           "decisions": 0, "gated": 0, "decision_ms": []}


def normalise(outcome):
    """one record shape across two demos that report different things.

    jev-ultrafast's runner counts `stories_visited` itself. browser_agent.py
    reports what it collected, keyed by url, which is the stronger measure and
    the one the task actually asks for.
    """
    if "stories_visited" not in outcome:
        if "collected" in outcome:
            # a demo that HAS a ledger is scored on it, even when it is empty.
            # falling through to the witness scored three dead runs as 1,
            # because the required pattern was on the front page they never
            # left.
            collected = outcome["collected"] or {}
            outcome["stories_visited"] = sum(1 for i in collected.values() if i)
        else:
            # the android agent collects nothing, so its score is the witness:
            # requirements that actually appeared on a screen it reached. a
            # demo with no ledger would otherwise read as zero for every model
            # and the sweep would rank nothing.
            outcome["stories_visited"] = (outcome.get("witness") or {}).get(
                "witnessed", 0)
        outcome["actions"] = outcome.get("steps", 0)
    outcome.setdefault("decisions", len(outcome.get("decision_ms", [])))
    for key, default in CRASHED.items():
        outcome.setdefault(key, default)
    return outcome


def one_run(python, demo, passthrough, env, before_each=None):
    if before_each:
        # a run that STARTS on the answer screen has not succeeded, it has been
        # handed the answer. android settings resumes wherever it was left, so
        # without a reset between runs the second onward measure nothing. the
        # reset is a caller's command because what "fresh" means is the
        # caller's business, not this harness's.
        #
        # and a reset that FAILED must stop the run. this used to be
        # check=False with the output captured and dropped, so a reset that
        # detected a bad device said so into a void and the run was scored as
        # a model result anyway.
        ready = subprocess.run(before_each, shell=True, env=env, check=False,
                               capture_output=True, text=True)
        if ready.returncode != 0:
            why = (ready.stderr or ready.stdout or "").strip().splitlines()
            return {"status": "reset-failed", "stories_visited": 0, "actions": 0,
                    "decisions": 0, "gated": 0, "decision_ms": [], "wall_s": 0.0,
                    "error": why[-1] if why else f"exit {ready.returncode}"}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        summary = f.name
    cmd = [python, demo, "--summary", summary, *passthrough]
    started = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    wall = time.monotonic() - started
    try:
        with open(summary) as f:
            outcome = normalise(json.load(f))
    except (OSError, json.JSONDecodeError):
        outcome = dict(CRASHED,
                       error=proc.stderr.strip().splitlines()[-1:] or ["no summary"])
    finally:
        os.unlink(summary)
    outcome["wall_s"] = round(wall, 1)
    return outcome


def report(runs, required):
    print(f"\n{len(runs)} runs\n")
    print(f"  {'run':>4} {'stories':>9} {'actions':>8} {'decisions':>10} "
          f"{'gated':>6} {'status':>9} {'wall':>7}")
    for i, r in enumerate(runs, 1):
        print(f"  {i:>4} {r['stories_visited']:>5}/{required:<3} {r['actions']:>8} "
              f"{r['decisions']:>10} {r['gated']:>6} {r['status']:>9} "
              f"{r['wall_s']:>6.0f}s")

    visited = [r["stories_visited"] for r in runs]
    complete = sum(1 for v in visited if v >= required)
    reached_one = sum(1 for v in visited if v >= 1)
    all_ms = [ms for r in runs for ms in r.get("decision_ms", [])]

    print(f"\n  task complete ({required}/{required} stories): "
          f"{complete}/{len(runs)}")
    print(f"  reached at least one story:      {reached_one}/{len(runs)}")
    print(f"  stories per run:                 "
          f"mean {statistics.mean(visited):.1f}, median {statistics.median(visited)}, "
          f"range {min(visited)}-{max(visited)}")
    if all_ms:
        print(f"  decision latency:                "
              f"median {statistics.median(all_ms):.0f} ms over {len(all_ms)} decisions")
    statuses = {}
    for r in runs:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"  final status:                    {statuses}")
    return {"runs": len(runs), "complete": complete, "reached_one": reached_one,
            "mean_stories": statistics.mean(visited), "statuses": statuses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--required", type=int, default=3)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--cdp", default="http://127.0.0.1:9222")
    ap.add_argument("--out", default=None, help="write raw outcomes here")
    ap.add_argument("--before-each", default=None, metavar="CMD",
                    help="shell command to run before every run. use it to put "
                         "the device back to a state that has not been handed "
                         "the answer")
    ap.add_argument("--demo", default="run_hn_demo.py",
                    help="which demo to repeat, relative to demos/")
    ap.add_argument("passthrough", nargs="*", help="args for the demo, after --")
    args = ap.parse_args()

    demo = os.path.join(HERE, args.demo)
    env = dict(os.environ, BU_CDP_URL=args.cdp)
    runs = []
    for i in range(args.runs):
        print(f"=== run {i + 1}/{args.runs}", flush=True)
        outcome = one_run(args.python, demo, args.passthrough, env,
                          args.before_each)
        runs.append(outcome)
        print(f"    {outcome['stories_visited']}/{args.required} stories, "
              f"{outcome['actions']} actions, {outcome['status']}", flush=True)

    summary = report(runs, args.required)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "runs": runs}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
