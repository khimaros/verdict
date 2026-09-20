#!/usr/bin/env python3
"""phase 0 probe: verify what llama-server actually does before we build on it.

stdlib only, so it can be dropped on any machine that can reach the server.
writes a json report and prints a human readable summary.

usage:
  ./scripts/probe_llama_server.py --base-url http://host:port --model qwen3.5-0.8b:Q8_0
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

# labels we intend to use for options; must be single tokens (spec section 4.2)
LABELS = list("ABCDEFGH")

# n_probs values to sweep when finding the server's cap
N_PROBS_SWEEP = [1, 10, 20, 100, 1000, 100000]

# a prefix long enough that a cache hit is unmistakable in the timings
LONG_PREFIX_REPEATS = 200

HTTP_TIMEOUT = 600


class Server:
    """thin http client over the native llama-server endpoints.

    llama-swap proxies them under /upstream/<model>/ and does not expose them
    at the root, so the prefix is resolved once at construction.
    """

    def __init__(self, base_url, model, upstream):
        self.base = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.upstream = upstream

    def url(self, path):
        if self.upstream and self.model:
            return f"{self.base}/upstream/{self.model}{path}"
        return f"{self.base}{path}"

    def post(self, path, payload):
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url(path), data=body, headers={"content-type": "application/json"}
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            out = json.loads(resp.read())
        return out, (time.monotonic() - t0) * 1000.0

    def get(self, path):
        t0 = time.monotonic()
        with urllib.request.urlopen(self.url(path), timeout=HTTP_TIMEOUT) as resp:
            out = json.loads(resp.read())
        return out, (time.monotonic() - t0) * 1000.0


def probe(fn):
    """run a check and capture failures instead of aborting the whole probe."""
    try:
        return {"ok": True, "result": fn()}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"http {e.code}: {e.read()[:400].decode(errors='replace')}"}
    # a probe reports what happened; it does not handle it
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def shape(obj, depth=0):
    """describe json structure without dumping megabytes of it."""
    if isinstance(obj, dict):
        if depth >= 3:
            return sorted(obj.keys())
        return {k: shape(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [f"list[{len(obj)}]", shape(obj[0], depth + 1) if obj else None]
    return type(obj).__name__


def completion(srv, **kw):
    payload = {"n_predict": 1, "cache_prompt": True, "temperature": -1.0}
    payload.update(kw)
    return srv.post("/completion", payload)


def entries_of(resp):
    """pull the per-token candidate list out of whatever shape this build uses."""
    probs = resp.get("completion_probabilities")
    if not probs:
        return None, None
    first = probs[0]
    for key in ("top_logprobs", "probs", "top_probs"):
        if key in first:
            return key, first[key]
    return None, first


def as_prob(entry):
    if "prob" in entry:
        return entry["prob"]
    if "logprob" in entry:
        return math.exp(entry["logprob"])
    return None


def token_of(entry):
    for key in ("token", "tok_str", "text_offset"):
        if key in entry:
            return entry[key]
    return None


def check_endpoints(srv):
    out = {}
    props, _ = srv.get("/props")
    out["build_info"] = props.get("build_info")
    out["model_path"] = props.get("model_path")
    out["total_slots"] = props.get("total_slots")
    out["n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
    tok, _ = srv.post("/tokenize", {"content": "hello world"})
    out["tokenize"] = tok
    tmpl, _ = srv.post(
        "/apply-template",
        {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]},
    )
    out["apply_template"] = tmpl
    return out


def check_labels(srv):
    """every option label must be exactly one token, or the whole scheme breaks."""
    out = {}
    for label in LABELS:
        tok, _ = srv.post("/tokenize", {"content": label, "add_special": False})
        out[label] = tok.get("tokens")
    return out


def check_completion_shape(srv):
    resp, ms = completion(srv, prompt="The capital of France is", n_probs=10)
    key, entries = entries_of(resp)
    return {
        "response_keys": sorted(resp.keys()),
        "probs_field": "completion_probabilities",
        "entry_field": key,
        "entry_shape": shape(entries[0]) if entries else None,
        "sample_entries": entries[:5] if entries else None,
        "timings": resp.get("timings"),
        "wall_ms": round(ms, 1),
    }


def check_token_prompt(srv):
    """section 4.2 depends on sending token ids, not text."""
    text = "The capital of France is"
    tok, _ = srv.post("/tokenize", {"content": text})
    ids = tok["tokens"]
    a, _ = completion(srv, prompt=text, n_probs=5)
    b, _ = completion(srv, prompt=ids, n_probs=5)
    _, ea = entries_of(a)
    _, eb = entries_of(b)
    return {
        "n_tokens": len(ids),
        "text_top": token_of(ea[0]) if ea else None,
        "ids_top": token_of(eb[0]) if eb else None,
        "text_prob": as_prob(ea[0]) if ea else None,
        "ids_prob": as_prob(eb[0]) if eb else None,
        "agree": bool(ea and eb and token_of(ea[0]) == token_of(eb[0])),
    }


def check_temperature(srv):
    """the readme claims temperature < 0 gives a plain softmax of the logits."""
    prompt = "The capital of France is"
    out = {}
    for name, kw in {
        "neg": {"temperature": -1.0},
        "zero": {"temperature": 0.0},
        "one_topk0": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0},
        "half": {"temperature": 0.5, "top_k": 0, "top_p": 1.0, "min_p": 0.0},
        "neg_post": {"temperature": -1.0, "post_sampling_probs": True},
    }.items():
        resp, _ = completion(srv, prompt=prompt, n_probs=5, **kw)
        _, e = entries_of(resp)
        out[name] = {
            "top": token_of(e[0]) if e else None,
            "probs": [round(as_prob(x), 6) for x in e[:5]] if e else None,
        }
    return out


def check_n_probs_cap(srv):
    out = {}
    for n in N_PROBS_SWEEP:
        try:
            resp, _ = completion(srv, prompt="The capital of France is", n_probs=n)
            _, e = entries_of(resp)
            out[str(n)] = len(e) if e else 0
        except urllib.error.HTTPError as err:
            out[str(n)] = f"http {err.code}"
    return out


def check_grammar(srv):
    """does constraining output to the labels change the reported probabilities?"""
    prompt = "Answer with one letter.\nQ: Is the sky blue? A) yes B) no\nAnswer:"
    plain, _ = completion(srv, prompt=prompt, n_probs=20)
    gram, _ = completion(srv, prompt=prompt, n_probs=20, grammar='root ::= "A" | "B"')
    _, ep = entries_of(plain)
    _, eg = entries_of(gram)

    def pick(entries):
        return {
            token_of(x): round(as_prob(x), 6)
            for x in (entries or [])
            if str(token_of(x)).strip() in ("A", "B")
        }

    return {
        "plain_top": token_of(ep[0]) if ep else None,
        "plain_labels": pick(ep),
        "grammar_top": token_of(eg[0]) if eg else None,
        "grammar_labels": pick(eg),
        "grammar_entries": len(eg) if eg else 0,
    }


def check_prefix_reuse(srv):
    """the whole performance story rests on the prefix staying cached."""
    filler = (
        "The customer reported an issue with their account and the agent "
        "escalated it to the relevant team for further review. "
    ) * LONG_PREFIX_REPEATS
    prefix = "System: you classify support tickets.\n\nTicket:\n" + filler + "\n---\n"
    runs = []
    for label, suffix, cache in [
        ("cold", "Question one. Answer:", True),
        ("warm_same", "Question one. Answer:", True),
        ("warm_new_suffix", "Question two, a different one. Answer:", True),
        ("nocache_default", "Question three. Answer:", None),
    ]:
        kw = {"prompt": prefix + suffix, "n_probs": 5}
        if cache is not None:
            kw["cache_prompt"] = cache
        else:
            kw["cache_prompt"] = None
        payload = {k: v for k, v in kw.items() if v is not None}
        resp, ms = completion(srv, **payload)
        t = resp.get("timings", {})
        runs.append(
            {
                "run": label,
                "sent_cache_prompt": cache,
                "wall_ms": round(ms, 1),
                "prompt_n": t.get("prompt_n"),
                "prompt_ms": round(t.get("prompt_ms", 0), 1),
                "tokens_cached": resp.get("tokens_cached"),
            }
        )
    return runs


def check_cache_determinism(srv):
    """llama.cpp warns logits are not bit-identical across batch shapes."""
    prompt = "Rate the urgency of this ticket. A) low B) high\nAnswer:"
    cold, _ = completion(srv, prompt=prompt, n_probs=5, cache_prompt=False)
    warm, _ = completion(srv, prompt=prompt, n_probs=5, cache_prompt=True)
    _, ec = entries_of(cold)
    _, ew = entries_of(warm)
    return {
        "cold": [(token_of(x), as_prob(x)) for x in (ec or [])[:3]],
        "warm": [(token_of(x), as_prob(x)) for x in (ew or [])[:3]],
        "argmax_agree": bool(ec and ew and token_of(ec[0]) == token_of(ew[0])),
        "bit_identical": bool(
            ec and ew and [as_prob(x) for x in ec[:3]] == [as_prob(x) for x in ew[:3]]
        ),
    }


CHECKS = [
    ("endpoints", check_endpoints),
    ("labels_single_token", check_labels),
    ("completion_shape", check_completion_shape),
    ("token_id_prompt", check_token_prompt),
    ("temperature", check_temperature),
    ("n_probs_cap", check_n_probs_cap),
    ("grammar", check_grammar),
    ("prefix_reuse", check_prefix_reuse),
    ("cache_determinism", check_cache_determinism),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default=None, help="model id, required behind llama-swap")
    ap.add_argument(
        "--upstream",
        action="store_true",
        help="route native endpoints via /upstream/<model>/ (llama-swap)",
    )
    ap.add_argument("--out", default="/tmp/probe_report.json")
    args = ap.parse_args()

    srv = Server(args.base_url, args.model, args.upstream)
    report = {"base_url": srv.base, "model": args.model, "upstream": args.upstream}
    for name, fn in CHECKS:
        print(f"== {name} ...", flush=True)
        report[name] = probe(lambda fn=fn: fn(srv))
        print(json.dumps(report[name], indent=2, default=str), flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0 if all(report[n]["ok"] for n, _ in CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
