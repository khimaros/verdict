#!/usr/bin/env python3
"""run a demo's reliability suite against every model, one at a time.

a verdict server binds one model, and llama-swap serves one at a time, so this
is necessarily sequential: start a server, wait for it to derive and answer,
run the suite, stop it, move on. a model that cannot be derived is recorded as
a refusal and the sweep continues -- a refusal is a result.

the per-model formatter is derived at runtime, so adding a model to the list is
the whole cost of adding it to the sweep.

usage:
  scripts/sweep_models.py --demo browser_agent.py --runs 3 \
      --models qwen3.5-9b:Q8_0 qwen3.5-4b:Q8_0 -- --goal "..." --require "..."
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "python"))

from llama_verdict import config

# openings for formats whose generation prompt ends before content begins. see
# docs/DECISIONS.md -- these are not model deficiencies, they are templates
# that stop early, and the list is short on purpose.
OVERRIDES = {"gpt-oss-20b:Q8_0": "<|start|>assistant<|channel|>final<|message|>",
             "gpt-oss-120b:Q8_0": "<|start|>assistant<|channel|>final<|message|>"}


def wait_until_ready(port, deadline):
    """a first derivation resolves 52 labels one call at a time, so the server
    can take a minute to answer. poll rather than guess."""
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def serve(model, base_url, port, log):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    cmd = [sys.executable, "-m", "llama_verdict.server", "--base-url", base_url,
           "--model", model, "--port", str(port), "--quiet"]
    if model in OVERRIDES:
        cmd += ["--assistant-open", OVERRIDES[model]]
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log)


def run_suite(demo, runs, port, passthrough, out_path, before_each=None):
    cmd = [sys.executable, os.path.join(ROOT, "demos", "reliability.py"),
           "--demo", demo, "--runs", str(runs), "--out", out_path]
    if before_each:
        cmd += ["--before-each", before_each]
    cmd += ["--", "--verdict", f"http://127.0.0.1:{port}", *passthrough]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout + proc.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", default="browser_agent.py")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=3)
    config.backend_url_argument(ap)
    ap.add_argument("--port", type=int, default=8495)
    ap.add_argument("--out-dir", default="eval/results/sweep")
    ap.add_argument("--startup-timeout", type=int, default=300)
    ap.add_argument("--before-each", default=None, metavar="CMD",
                    help="shell command run before every run, to put the device "
                         "back to a state that has not been handed the answer")
    ap.add_argument("passthrough", nargs="*")
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    summary = {}

    for model in args.models:
        slug = model.replace(":", "_").replace("/", "_")
        print(f"\n=== {model}", flush=True)
        log_path = os.path.join("/tmp", f"sweep-server-{slug}.log")
        with open(log_path, "w") as log:
            server = serve(model, args.base_url, args.port, log)
            try:
                started = time.monotonic()
                if not wait_until_ready(args.port,
                                        started + args.startup_timeout):
                    server.terminate()
                    with open(log_path) as f:
                        why = f.read().strip().splitlines()[-1:] or ["no output"]
                    print(f"    server never came up: {why[0][:120]}", flush=True)
                    summary[model] = {"status": "server-failed", "detail": why[0]}
                    continue
                print(f"    ready in {time.monotonic() - started:.0f}s", flush=True)

                out_path = os.path.join(ROOT, args.out_dir, f"{slug}.json")
                text = run_suite(args.demo, args.runs, args.port,
                                 args.passthrough, out_path,
                                 args.before_each)
                for line in text.splitlines():
                    if "stories," in line or "task complete" in line:
                        print(f"    {line.strip()}", flush=True)
                try:
                    with open(out_path) as f:
                        summary[model] = json.load(f)["summary"]
                except (OSError, json.JSONDecodeError, KeyError):
                    summary[model] = {"status": "no-summary"}
            finally:
                server.terminate()
                server.wait(timeout=30)

    path = os.path.join(ROOT, args.out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
