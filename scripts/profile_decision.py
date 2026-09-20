#!/usr/bin/env python3
"""where does the time in one decision actually go?

replays a captured jev request through the real decide() path and accounts for
every http call. optimising before running this is guessing.

usage:
  ./scripts/profile_decision.py --base-url http://host:port --model M \
      --formatter spec/formatters/M.json --request /tmp/request-001.json
"""

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from llama_verdict.backend import HttpBackend
from llama_verdict.decide import Decider
from llama_verdict.types import Formatter


class Timed(HttpBackend):
    """records every round trip so the total can be attributed rather than guessed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = []

    def _post(self, path, payload):
        t0 = time.monotonic()
        out, ms = super()._post(path, payload)
        self.calls.append({
            "path": path,
            "ms": (time.monotonic() - t0) * 1000.0,
            "n_probs": payload.get("n_probs"),
            "prompt_tokens": (len(payload["prompt"])
                              if isinstance(payload.get("prompt"), list) else 0),
            "evaluated": out.get("timings", {}).get("prompt_n"),
            "cached": out.get("tokens_cached"),
        })
        return out, ms


def report(backend, wall):
    by_path = collections.defaultdict(lambda: {"n": 0, "ms": 0.0})
    for c in backend.calls:
        entry = by_path[c["path"]]
        entry["n"] += 1
        entry["ms"] += c["ms"]

    print(f"\nwall {wall:.0f} ms over {len(backend.calls)} round trips\n")
    print(f"  {'endpoint':<14} {'calls':>6} {'total ms':>10} {'share':>7}")
    for path, e in sorted(by_path.items(), key=lambda kv: -kv[1]["ms"]):
        print(f"  {path:<14} {e['n']:>6} {e['ms']:>10.0f} {100 * e['ms'] / wall:>6.1f}%")
    accounted = sum(e["ms"] for e in by_path.values())
    print(f"  {'unaccounted':<14} {'':>6} {wall - accounted:>10.0f} "
          f"{100 * (wall - accounted) / wall:>6.1f}%")

    scores = [c for c in backend.calls if c["path"] == "/completion"]
    if scores:
        print(f"\n  {'#':>3} {'ms':>7} {'sent':>7} {'evaluated':>10} {'cached':>7} {'n_probs':>8}")
        for i, c in enumerate(scores, 1):
            print(f"  {i:>3} {c['ms']:>7.0f} {c['prompt_tokens']:>7} "
                  f"{c['evaluated'] or 0:>10} {c['cached'] or 0:>7} {c['n_probs']:>8}")
        # a text prompt has no client-side token count, so measure reuse
        # against what the server reports it held rather than what we sent
        reused = sum(1 for c in scores
                     if (c["evaluated"] or 0) < (c["cached"] or 0) * 0.5)
        print(f"\n  prefix reuse: {reused}/{len(scores)} calls evaluated under half "
              f"the prompt the server held")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--formatter", required=True)
    ap.add_argument("--request", required=True, help="a captured jev request body")
    ap.add_argument("--tournament", action="store_true")
    args = ap.parse_args()

    with open(args.request) as f:
        body = json.load(f)

    backend = Timed(args.base_url, args.model)
    decider = Decider(backend, Formatter.load(args.formatter), tournament=args.tournament)

    sizes = {n: len(q.get("criteria", {})) for n, q in body["questions"].items()}
    print(f"questions: {sizes}")

    t0 = time.monotonic()
    result = decider.decide(body["state"], body["questions"])
    wall = (time.monotonic() - t0) * 1000.0

    for name, a in result["answers"].items():
        cache = a.get("cache", {})
        print(f"  {name:<18} -> {a['top']!s:<8} conf={a['confidence']:.3f} "
              f"mass={a['option_mass']:.4f} retries={cache.get('retries', '-')}")
    report(backend, wall)


if __name__ == "__main__":
    sys.exit(main())
