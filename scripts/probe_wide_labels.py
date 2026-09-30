#!/usr/bin/env python3
"""is an unfamiliar label as usable as `A`?

the wide alphabet (spec section 10.1) lets a list past 52 options be read in
one exact pass, by labelling it with single-character tokens out of the model's
own vocabulary. the labels are verified single tokens at the scored position.
that is an encoding property and says nothing about whether the model will
REACH for one.

measuring that at 104 options answers nothing, because length and alphabet
change together and spec 10.3 already says a long list is a quality problem on
its own. so the question is asked with LENGTH HELD FIXED: the same cases, the
same distractors, the same positions, scored under each alphabet, differing
only in which characters label the options.

  ascii     the pinned A-Za-z, what every short list uses today
  wide      non-ascii single-character tokens only, same count, same order
  distinct  the same, minus anything confusable with an ascii letter

a gap between the first two columns is the alphabet. no gap means an
unfamiliar label costs nothing and the wide alphabet is free above 52 too.

the third column tested a specific suspect and REFUTED it. ordering candidates
by token id puts the commonest non-ascii characters first, and those turn out
to be accented latin and cyrillic: the alphabet begins `e`-acute, then cyrillic
o, a, e, t, and **cyrillic o and latin o are different tokens that look
identical.** a model shown `o) Site logo` cannot see which was meant, so
answering with the latin token would drop the mass while leaving the ranking
intact -- exactly the symptom.

dropping the lookalikes makes it WORSE. qwen3.5-9b goes from 16/16 at mass
0.7780 to 11/16 at 0.3146, and the readout has to widen to the whole
vocabulary to find the labels.

frequency looked like the reason -- the lookalikes being the commoner
characters -- and `scripts/probe_label_rates.py` measured that directly and
refuted it: the dropped-lookalike set has a HIGHER base rate than the set that
scores better. so why it hurts is open. **this filter is a measuring
instrument and not a fix, and must not move into `derive.label_candidates`.**

the long-list row is reported separately and is NOT evidence about the
alphabet, since nothing there is held fixed.

usage:
  scripts/probe_wide_labels.py --models qwen3.5-9b:Q8_0 minicpm5-2b:Q8_0
"""

import argparse
import json
import os
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

from eval_interventions import POSITIONS, STATE, case
from llama_verdict import config, derive, extract, prompt, spec, types
from llama_verdict.backend import HttpBackend
from smoke_models import OVERRIDES

# enough labels that the wide-only column never has to borrow an ascii one,
# and enough left over for the confusable-free column after the lookalikes are
# dropped, which on the models measured removes most of the first few hundred
HEADROOM = 600
COUNTS = [8, 16, 26, 52]
LONG = 104

# the pinned alphabet is latin, so every OTHER latin letter is a lookalike by
# construction: `l` with stroke and dotless `i` are not decompositions and are
# still read as `l` and `i`. cyrillic and greek earn their place separately --
# cyrillic o/a/e/c/p/x and greek omicron/alpha are drawn identically to latin
# and do not decompose at all.
CONFUSABLE_SCRIPTS = ("LATIN", "CYRILLIC", "GREEK")


def confusable(char):
    """would a reader mistake this character for an ascii letter?

    two mechanisms, neither needing a confusables table. an accented or
    compatibility form DECOMPOSES to an ascii base under NFKD, and everything
    else is caught by script: a label drawn from latin, cyrillic or greek is
    competing with `A`-`Za`-`z` for the same glyph.

    a label a reader cannot distinguish is one the MODEL cannot distinguish
    either, and answering with the latin token it thinks it sees puts the mass
    outside every declared option -- which is what a drop in option mass with
    accuracy intact looks like.
    """
    base = unicodedata.normalize("NFKD", char)[0]
    if base.isascii() and base.isalpha():
        return True
    name = unicodedata.name(char, "")
    return any(name.startswith(script) for script in CONFUSABLE_SCRIPTS)


def score_with(backend, formatter, question, alphabet, ids):
    """one pass at one position under a given alphabet.

    deliberately one layer below `Decider`: the point is to vary the alphabet
    freely, and the decider is what pins it to the spec's order.
    """
    labels = prompt.label_map(question, alphabet)
    label_ids = [ids[label] for label in labels.values()]
    t0 = time.monotonic()
    scored = backend.score(
        prompt.render_prefix(formatter, STATE)
        + prompt.render_suffix(formatter, question, alphabet), label_ids)
    raw = {oid: scored["raw"][i] for oid, i in zip(labels, label_ids, strict=True)}
    probs, mass = extract.renormalise(raw)
    top = max(probs, key=probs.get) if probs else None
    # the rung reached is its own result. an unfamiliar label is unlikely to
    # sit in the top 64 candidates, so the readout ladder escalates to find
    # it -- a latency cost that belongs to the alphabet and is invisible in
    # accuracy or in option mass.
    return {"ok": top == "correct", "option_mass": mass,
            "correct_prob": probs.get("correct", 0.0),
            "label": labels["correct"], "n_probs_used": scored["n_probs_used"],
            "ms": round((time.monotonic() - t0) * 1000.0)}


def probe(base_url, model, counts, long_count):
    backend = HttpBackend(base_url, model)
    started = time.monotonic()
    formatter, _ = derive.build(backend, model,
                                assistant_open=OVERRIDES.get(model))
    wide = derive.ensure_wide_labels(backend, formatter, HEADROOM)

    pinned = list(spec.constants()["labels"])
    non_ascii = [ell for ell in wide if not ell.isascii()]
    distinct = [ell for ell in non_ascii if not confusable(ell)]
    alphabets = {"ascii": pinned, "wide": non_ascii}
    # only offered when it can label the longest case without borrowing
    if len(distinct) >= max(counts):
        alphabets["distinct"] = distinct

    rows = []
    for name, alphabet in alphabets.items():
        for n in counts:
            for position in POSITIONS:
                q = types.parse_question("pick", case(n, position)["pick"])
                row = score_with(backend, formatter, q, alphabet, wide)
                rows.append({"alphabet": name, "n": n, "position": position, **row})

    # the shipped order, past the ceiling. reported, never compared: at this
    # length nothing is held fixed and the number is about the list, not the
    # labels.
    long_rows = []
    for position in POSITIONS:
        q = types.parse_question("pick", case(long_count, position)["pick"])
        row = score_with(backend, formatter, q, list(wide), wide)
        long_rows.append({"n": long_count, "position": position, **row})

    def tally(subset):
        return {"correct": f"{sum(r['ok'] for r in subset)}/{len(subset)}",
                "accuracy": sum(r["ok"] for r in subset) / len(subset),
                "min_option_mass": min(r["option_mass"] for r in subset),
                "max_n_probs": max(r.get("n_probs_used", 0) for r in subset),
                "median_ms": sorted(r.get("ms", 0) for r in subset)[len(subset) // 2]}

    return {
        "model": model,
        "wide_labels_available": len(wide),
        "non_ascii_available": len(non_ascii),
        "distinct_available": len(distinct),
        "first_non_ascii": non_ascii[0] if non_ascii else None,
        "first_distinct": distinct[0] if distinct else None,
        "by_alphabet": {name: tally([r for r in rows if r["alphabet"] == name])
                        for name in alphabets},
        "by_alphabet_count": {
            name: {n: f"{sum(r['ok'] for r in rows if r['alphabet'] == name and r['n'] == n)}/4"
                   for n in counts} for name in alphabets},
        "long_list": dict(tally(long_rows), n=long_count),
        "wall_s": round(time.monotonic() - started, 1),
        "rows": rows, "long_rows": long_rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    config.backend_url_argument(ap)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--counts", type=int, nargs="+", default=COUNTS)
    ap.add_argument("--long", type=int, default=LONG)
    ap.add_argument("--out", default="eval/results/wide-labels.json")
    args = ap.parse_args()

    report = []
    for model in args.models:
        try:
            row = probe(args.base_url, model, args.counts, args.long)
            cols = "  ".join(
                f"{name} {t['correct']:>6} ({t['min_option_mass']:.4f}, "
                f"{t['median_ms']}ms, n_probs {t['max_n_probs']})"
                for name, t in row["by_alphabet"].items())
            print(f"{model:<24} {cols}  "
                  f"n={row['long_list']['n']} {row['long_list']['correct']:>4}  "
                  f"labels {row['wide_labels_available']}/"
                  f"{row['distinct_available']}  {row['wall_s']:.0f}s",
                  flush=True)
        except Exception as e:
            row = {"model": model, "error": f"{type(e).__name__}: {e}"}
            print(f"{model:<24} FAILED  {row['error'][:80]}", flush=True)
        report.append(row)

    out = os.path.join(HERE, "..", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
