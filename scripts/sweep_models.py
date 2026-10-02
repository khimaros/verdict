#!/usr/bin/env python3
"""run a demo's reliability suite against every model, one at a time.

llama-swap serves one model at a time, so this is necessarily sequential: wait
for the endpoint to derive the model and answer for it, run the suite, move
on. a model that cannot be derived is recorded as a refusal and the sweep
continues -- a refusal is a result.

with --endpoint the suite runs through a verdict endpoint that is already up,
which routes each request to the model it names; without it a server is
started here for each model in turn. the per-model formatter is derived at
runtime either way, so adding a model to the list is the whole cost of adding
it to the sweep.

usage:
  scripts/sweep_models.py --demo browser_agent.py --runs 3 \
      --models qwen3.5-9b:Q8_0 qwen3.5-4b:Q8_0 -- --goal "..." --require "..."
"""

import argparse
import contextlib
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

WARM_UP = {"state": "warm up",
           "questions": {"q": {"type": "noul", "instructions": "is this a test?"}}}
# one request may be the model's first, which derives and verifies it
WARM_UP_TIMEOUT = 900
# the endpoint's answer for a model it cannot read: a result, not a wait
REFUSED = 422


def not_ready(url, model, deadline):
    """ask the endpoint one question of this model until it answers.

    a first derivation can take minutes, and a server started here may not be
    listening yet, so poll rather than guess. returns None once it answers,
    else why it never did; a model the endpoint refuses is not waited for.
    """
    request = urllib.request.Request(
        url + "/v1/systemone", json.dumps({"model": model, **WARM_UP}).encode(),
        {"content-type": "application/json"})
    why = "no answer"
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(request, timeout=WARM_UP_TIMEOUT)
            return None
        except urllib.error.HTTPError as e:
            why = f"{e.code} {e.read().decode(errors='replace')}"
            if e.code == REFUSED:
                return why
        except (urllib.error.URLError, OSError) as e:
            why = str(e)
        time.sleep(2)
    return why


@contextlib.contextmanager
def endpoint_for(model, args, log_path):
    """the url of a verdict endpoint that answers for `model`: the running one
    named with --endpoint, which routes each request to the model it names, or
    one started here for this model and stopped afterwards."""
    if args.endpoint:
        yield args.endpoint.rstrip("/")
        return
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    cmd = [sys.executable, "-m", "llama_verdict.server", "--base-url", args.base_url,
           "--model", model, "--port", str(args.port), "--quiet"]
    with open(log_path, "w") as log:
        server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            yield f"http://127.0.0.1:{args.port}"
        finally:
            server.terminate()
            server.wait(timeout=30)


def run_suite(demo, runs, url, model, passthrough, out_path, before_each=None, log=None):
    cmd = [sys.executable, os.path.join(ROOT, "demos", "reliability.py"),
           "--demo", demo, "--runs", str(runs), "--out", out_path]
    if log:
        cmd += ["--log", log]
    if before_each:
        cmd += ["--before-each", before_each]
    cmd += ["--", "--verdict", url, "--model", model, *passthrough]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout + proc.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", default="browser_agent.py")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--endpoint", default=config.get("VERDICT_ENDPOINT"),
                    help="a verdict endpoint that is already running, which routes "
                         "each request to the model it names. without it a server "
                         "is started here per model [VERDICT_ENDPOINT]")
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
        with endpoint_for(model, args, log_path) as url:
            started = time.monotonic()
            why = not_ready(url, model, started + args.startup_timeout)
            if why:
                print(f"    never answered: {why[:160]}", flush=True)
                summary[model] = {"status": "server-failed", "detail": why}
                continue
            print(f"    ready in {time.monotonic() - started:.0f}s", flush=True)

            out_path = os.path.join(ROOT, args.out_dir, f"{slug}.json")
            # every step as it happens. outside the repo: a run's own output
            # names the endpoint it was sent to
            steps = os.path.join("/tmp", f"sweep-steps-{slug}.log")
            print(f"    steps in {steps}", flush=True)
            text = run_suite(args.demo, args.runs, url, model, args.passthrough,
                             out_path, args.before_each, steps)
            for line in text.splitlines():
                if "stories," in line or "task complete" in line:
                    print(f"    {line.strip()}", flush=True)
            try:
                with open(out_path) as f:
                    summary[model] = json.load(f)["summary"]
            except (OSError, json.JSONDecodeError, KeyError):
                summary[model] = {"status": "no-summary"}

    path = os.path.join(ROOT, args.out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
