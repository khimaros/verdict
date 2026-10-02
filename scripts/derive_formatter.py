#!/usr/bin/env python3
"""derive a formatter from a model's own chat template, then verify it.

the affixes come out of the template mechanically: render it with sentinel
message content, then diff renders that differ by one thing to isolate each
affix. nothing here knows anything about a model family.

the derived formatter is then measured. a formatter that cannot steer the
model to the labels is not written out, because a silently wrong opening still
produces plausible looking answers. see docs/DECISIONS.md.

usage:
  ./scripts/derive_formatter.py            # backend and model from the .env
  ./scripts/derive_formatter.py --base-url http://host:port \
      --models qwen3.5-0.8b:Q8_0 gemma-4-e2b-it:Q8_0
"""

import argparse
import functools
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_prompt_layout as P
from llama_verdict import config
from llama_verdict import derive as D

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..", "spec")

# the derivation itself lives in the package, because it is the runtime path
# now (SPEC 4.3) and not a script-only step. this script stays as the way to
# refresh the golden fixtures in spec/formatters/ (SPEC 4.4).
SYS_SENTINEL = D.SYS_SENTINEL
USER_SENTINEL = D.USER_SENTINEL
environment = D.environment
render = D.render


@functools.lru_cache(maxsize=1)
def constants():
    with open(os.path.join(SPEC, "constants.json")) as f:
        return json.load(f)


derive = D.affixes_from_template


def as_family(affixes):
    """affixes in the shape the probe's prompt builder takes."""
    def esc(s):
        return s.replace("{", "{{").replace("}", "}}")
    return {"system": esc(affixes["system_open"]) + "{sys}" + esc(affixes["system_close"]),
            "user_open": affixes["user_open"], "user_close": affixes["user_close"],
            "assistant_open": affixes["assistant_open"]}


def verify(cli, affixes):
    """measure the derived formatter before trusting it."""
    fam = as_family(affixes)
    masses, correct, splits = [], 0, []
    for state, question, options, expected in P.CASES:
        r = P.score(cli, fam, state, question, options)
        masses.append(r["option_mass"])
        correct += r["top"] == expected
        splits.append(r["split_clean"])
    return {"cases": len(P.CASES),
            "mean_option_mass": sum(masses) / len(masses),
            "min_option_mass": min(masses),
            "smoke_correct": f"{correct}/{len(P.CASES)}",
            "split_clean": all(splits)}


def label_ids(cli, affixes, n=52):
    """every label a single token, and still its own token after the opening.

    the ids are returned so they can be pinned into the artifact. they are a
    per-model constant, and resolving them at runtime costs one http round trip
    per label: 26 of the 29 tokenize calls in a profiled two-question decision.
    """
    opening = affixes["assistant_open"]
    open_tok = cli.tokenize(opening)
    ids = {}
    for ell in constants()["labels"][:n]:
        tok = cli.tokenize(ell)
        if len(tok) != 1 or cli.tokenize(opening + ell) != open_tok + tok:
            return None, ell
        ids[ell] = tok[0]
    return ids, None


def run(base_url, model, floor):
    cli = P.Client(base_url, model)
    props = cli.get("/props")
    template = props["chat_template"]
    affixes = derive(template, props.get("bos_token", ""), props.get("eos_token", ""))
    # pinned formatters are chat layout fixtures, which have a system turn
    del affixes["bare_user_open"]

    ids, bad = label_ids(cli, affixes)
    if not ids:
        raise SystemExit(f"{model}: label {bad!r} is not usable at the assistant position")

    v = verify(cli, affixes)
    if v["mean_option_mass"] < floor:
        raise SystemExit(
            f"{model}: mean option mass {v['mean_option_mass']:.4g} is below the "
            f"floor of {floor}. the derived opening does not steer this model; "
            f"refusing to write a formatter.")

    return {
        "spec_version": constants()["spec_version"],
        "model": model,
        "model_alias": props.get("model_alias"),
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "derived_from": "llama-server /props chat_template",
        "affixes": affixes,
        "label_ids": ids,
        "verification": dict(v, build_info=props.get("build_info"),
                             labels_single_token=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=config.backend_url(),
                    help="[LLAMA_VERDICT_URL]")
    ap.add_argument("--models", nargs="+",
                    default=config.backend_model() and [config.backend_model()],
                    help="[LLAMA_VERDICT_MODEL]")
    ap.add_argument("--floor", type=float, default=None)
    args = ap.parse_args()
    if not args.base_url or not args.models:
        ap.error("requires a backend and a model: pass --base-url and --models, "
                 "or set LLAMA_VERDICT_URL and LLAMA_VERDICT_MODEL in the "
                 "environment or a .env (see .env.example)")

    floor = (args.floor if args.floor is not None
             else constants()["option_mass_floor_formatter"])
    outdir = os.path.join(SPEC, "formatters")
    os.makedirs(outdir, exist_ok=True)

    for model in args.models:
        print(f"== {model}", flush=True)
        formatter = run(args.base_url, model, floor)
        path = os.path.join(outdir, model.replace(":", "_").replace("/", "_") + ".json")
        with open(path, "w") as f:
            json.dump(formatter, f, indent=2)
            f.write("\n")
        print(json.dumps(formatter["affixes"], indent=2), flush=True)
        print(json.dumps(formatter["verification"], indent=2), flush=True)
        print(f"wrote {path}\n", flush=True)


if __name__ == "__main__":
    sys.exit(main())
