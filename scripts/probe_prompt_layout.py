#!/usr/bin/env python3
"""verify the decision prompt layout against several models.

checks the things that must hold before a model family is usable:
single token labels, no merge between the assistant header and the label,
a prefix/suffix boundary that tokenises the same split as joint, and enough
signal on hand-picked cases to show the mechanism is engaged.

this is not an evaluation. accuracy here is a smoke test, not a number to quote.

usage:
  ./scripts/probe_prompt_layout.py --base-url http://host:port \
      --models qwen3.5-0.8b:Q8_0 gemma-4-e2b-it:Q8_0
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

SYSTEM = (
    "You are a precise classifier. Read the material, then answer the question "
    "with exactly one option label. Reply with the label only."
)

LABELS = list("ABCDEFGH")

# widen the readout until every label is present. the last rung is past any
# current vocabulary, so it returns the whole distribution and settles it.
N_PROBS_LADDER = [64, 4096, 300000]

STATE_DELIMITER = "\n---\n\n"
ANSWER_INSTRUCTION = "\n\nAnswer with one label."

# hand written chat formatters. the server's /apply-template cannot be used:
# for qwen3.5 it emits the thinking form of the assistant turn, which would put
# the label inside an open <think> block. see docs/DECISIONS.md.
FAMILIES = {
    "qwen3.5": {
        "system": "<|im_start|>system\n{sys}<|im_end|>\n",
        "user_open": "<|im_start|>user\n",
        "user_close": "<|im_end|>\n",
        "assistant_open": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
    "gemma4": {
        "system": "<bos><|turn>system\n{sys}<turn|>\n",
        "user_open": "<|turn>user\n",
        "user_close": "<turn|>\n",
        "assistant_open": "<|turn>model\n",
    },
}

# model id prefix -> family. longest match wins.
FAMILY_BY_PREFIX = {"qwen3.5": "qwen3.5", "qwen3.6": "qwen3.5", "qwen3.8": "qwen3.5",
                    "gemma-4": "gemma4"}

ROUTING = [("access", "Account access support: logins, passwords, permissions."),
           ("billing", "Billing support: invoices, payments, refunds."),
           ("technical", "Technical support: bugs, outages, integrations.")]
URGENCY = [("low", "Not urgent."), ("med", "Should be handled soon."),
           ("high", "Blocking, handle now.")]
YESNO = [("yes", "Yes."), ("no", "No.")]

CASES = [
    ("Customer cannot access an account after a password reset.",
     "Which queue should handle this request?", ROUTING, "access"),
    ("My invoice charged me twice for the same month and I want a refund.",
     "Which queue should handle this request?", ROUTING, "billing"),
    ("The export button throws a 500 error on every browser we tried.",
     "Which queue should handle this request?", ROUTING, "technical"),
    ("I forgot which email address my login is under.",
     "Which queue should handle this request?", ROUTING, "access"),
    ("This is the third outage this week and we are losing sales right now.",
     "How urgent is this message?", URGENCY, "high"),
    ("Just writing to say the new dashboard looks nice. No rush at all.",
     "How urgent is this message?", URGENCY, "low"),
    ("The report is wrong but we can work around it until next week.",
     "How urgent is this message?", URGENCY, "med"),
    ("If this is not fixed today I am cancelling our subscription.",
     "Does the sender threaten to stop being a customer?", YESNO, "yes"),
    ("Thanks for the quick fix, everything works now.",
     "Does the sender threaten to stop being a customer?", YESNO, "no"),
    ("Thanks for the quick fix, everything works now.",
     "Is the sender expressing satisfaction?", YESNO, "yes"),
]

HTTP_TIMEOUT = 900


class Client:
    def __init__(self, base_url, model):
        base = base_url.rstrip("/").removesuffix("/v1")
        self.base = f"{base}/upstream/{model}"
        self.model = model

    def post(self, path, payload):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return json.loads(r.read()), (time.monotonic() - t0) * 1000.0

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=HTTP_TIMEOUT) as r:
            return json.loads(r.read())

    def tokenize(self, text, add_special=False):
        out, _ = self.post("/tokenize", {"content": text, "add_special": add_special,
                                         "parse_special": True})
        return out["tokens"]


def family_for(model):
    match = max((p for p in FAMILY_BY_PREFIX if model.startswith(p)), key=len, default=None)
    if not match:
        raise SystemExit(f"no chat formatter known for model {model}")
    return FAMILIES[FAMILY_BY_PREFIX[match]]


def build(fam, state, question, options):
    """everything shared across questions goes in the prefix (spec 4.2)."""
    prefix = fam["system"].format(sys=SYSTEM) + fam["user_open"] + state + STATE_DELIMITER
    lines = "\n".join(f"{LABELS[i]}) {desc}" for i, (_, desc) in enumerate(options))
    suffix = (question + "\n" + lines + ANSWER_INSTRUCTION
              + fam["user_close"] + fam["assistant_open"])
    return prefix, suffix


def score(cli, fam, state, question, options):
    """a label missing from the top-n is a truncation artifact, not a zero.

    retrying at a wider n_probs is what separates "the model gives this option
    no mass" from "we did not look far enough down the list". without it the
    renormalised distribution is computed over whichever labels happened to
    survive, which looks like a confident answer and is not one.
    """
    prefix, suffix = build(fam, state, question, options)
    ptok, stok = cli.tokenize(prefix), cli.tokenize(suffix)
    label_ids = [cli.tokenize(LABELS[i])[0] for i in range(len(options))]

    wall, retries = 0.0, 0
    for n_probs in N_PROBS_LADDER:
        resp, ms = cli.post("/completion", {
            "prompt": ptok + stok, "n_predict": 1, "n_probs": n_probs,
            "cache_prompt": True, "post_sampling_probs": False, "temperature": -1.0})
        wall += ms
        entries = resp["completion_probabilities"][0]["top_logprobs"]
        by_id = {e["id"]: math.exp(e["logprob"]) for e in entries}
        missing = [i for i in label_ids if i not in by_id]
        returned = len(entries)
        # the server returns min(n_probs, vocab), so a short list means we have
        # the whole vocabulary and a still-missing label is genuinely absent
        if not missing or returned < n_probs:
            break
        retries += 1

    raw = {oid: by_id.get(label_ids[i], 0.0) for i, (oid, _) in enumerate(options)}
    mass = sum(raw.values())
    probs = {k: v / mass for k, v in raw.items()} if mass > 0 else raw
    top = max(probs, key=probs.get) if mass > 0 else None
    return {"probs": probs, "top": top, "option_mass": mass,
            "prompt_n": resp["timings"]["prompt_n"], "wall_ms": wall,
            "n_probs_used": n_probs, "retries": retries, "missing_labels": len(missing),
            "top_unconstrained": entries[0]["token"],
            "split_clean": ptok + stok == cli.tokenize(prefix + suffix)}


def check_structure(cli, fam):
    """the invariants that make the whole scheme work."""
    opening = fam["assistant_open"]
    open_tok = cli.tokenize(opening)
    label_tok = {ell: cli.tokenize(ell) for ell in LABELS}
    merges = [ell for ell in LABELS
              if cli.tokenize(opening + ell) != open_tok + label_tok[ell]]
    return {
        "labels_single_token": {k: v for k, v in label_tok.items() if len(v) != 1} or "all single",
        "label_ids": {k: v[0] for k, v in label_tok.items() if len(v) == 1},
        "assistant_open_tokens": open_tok,
        "labels_merging_with_header": merges or "none",
    }


def run_model(base_url, model):
    cli = Client(base_url, model)
    fam = family_for(model)
    props = cli.get("/props")
    out = {
        "model": model,
        "build_info": props.get("build_info"),
        "model_alias": props.get("model_alias"),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
        "total_slots": props.get("total_slots"),
        "structure": check_structure(cli, fam),
        "cases": [],
    }
    correct = 0
    masses = []
    for state, question, options, expected in CASES:
        r = score(cli, fam, state, question, options)
        ok = r["top"] == expected
        correct += ok
        masses.append(r["option_mass"])
        out["cases"].append({
            "state": state[:48], "expected": expected, "got": r["top"], "ok": ok,
            "probs": {k: round(v, 4) for k, v in r["probs"].items()},
            "option_mass": r["option_mass"], "n_probs_used": r["n_probs_used"],
            "missing_labels": r["missing_labels"],
            "top_unconstrained": r["top_unconstrained"],
            "split_clean": r["split_clean"], "wall_ms": round(r["wall_ms"], 1)})
    out["summary"] = {
        "correct": f"{correct}/{len(CASES)}",
        "min_option_mass": min(masses),
        "mean_option_mass": sum(masses) / len(masses),
        "needed_retry": sum(1 for c in out["cases"] if c["n_probs_used"] != N_PROBS_LADDER[0]),
        "all_splits_clean": all(c["split_clean"] for c in out["cases"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/layout_report.json")
    args = ap.parse_args()

    report = []
    for model in args.models:
        print(f"== {model} ...", flush=True)
        try:
            r = run_model(args.base_url, model)
        except (urllib.error.URLError, KeyError, SystemExit) as e:
            r = {"model": model, "error": f"{type(e).__name__}: {e}"}
        report.append(r)
        print(json.dumps(r.get("summary", r), indent=2), flush=True)
        print(json.dumps(r.get("structure", {}), indent=2), flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
