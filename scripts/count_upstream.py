#!/usr/bin/env python3
"""a recording proxy between verdict and llama-server.

answers the questions a latency number cannot: how many upstream calls does one
agent step make, at which n_probs rung, how big is each response, and how much
of a prompt is shared with the previous one.

it exists as a proxy rather than as instrumentation inside the client because
the client is the thing under measurement, and because a proxy records what was
actually sent rather than what the sender believed it sent.

usage:
  scripts/count_upstream.py --upstream http://10.1.200.250:7860 --port 8490 \
      --out /tmp/upstream.jsonl
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

# how much of a prompt to keep for the shared-prefix comparison. the whole
# thing would make the log larger than the thing it measures.
PREFIX_KEEP = 200000


def common_prefix(a, b):
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class Proxy(BaseHTTPRequestHandler):
    upstream = ""
    out = None
    previous = ""
    seen: ClassVar[dict] = {}

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        started = time.monotonic()
        request = urllib.request.Request(
            self.upstream + self.path, data=raw,
            headers={"content-type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read()
                status = response.status
        except urllib.error.HTTPError as e:
            body, status = e.read(), e.code
        elapsed = (time.monotonic() - started) * 1000

        self.record(raw, body, status, elapsed)

        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def record(self, raw, body, status, elapsed):
        try:
            sent = json.loads(raw)
        except json.JSONDecodeError:
            sent = {}
        prompt = sent.get("prompt")
        text = prompt if isinstance(prompt, str) else json.dumps(prompt)
        text = text or ""
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]

        entry = {
            "t": round(time.time(), 3),
            "path": self.path,
            "ms": round(elapsed),
            "status": status,
            "prompt_kind": type(prompt).__name__,
            "prompt_chars": len(text),
            "n_probs": sent.get("n_probs"),
            "response_bytes": len(body),
            "prompt_sha": digest,
            "duplicate": digest in Proxy.seen,
            "shared_prefix": common_prefix(text[:PREFIX_KEEP],
                                           Proxy.previous[:PREFIX_KEEP]),
        }
        Proxy.seen[digest] = Proxy.seen.get(digest, 0) + 1
        Proxy.previous = text
        Proxy.out.write(json.dumps(entry) + "\n")
        Proxy.out.flush()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", default="http://10.1.200.250:7860")
    ap.add_argument("--port", type=int, default=8490)
    ap.add_argument("--out", default="/tmp/upstream.jsonl")
    args = ap.parse_args()

    Proxy.upstream = args.upstream.rstrip("/")
    print(f"recording {args.upstream} -> {args.out} on port {args.port}",
          file=sys.stderr)
    with open(args.out, "w") as out:
        Proxy.out = out
        ThreadingHTTPServer(("127.0.0.1", args.port), Proxy).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
