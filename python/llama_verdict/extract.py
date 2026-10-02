"""turn a next-token distribution into a typed answer.

spec/SPEC.md sections 6 and 7 are normative.
"""

import math
import typing

from . import spec


def renormalise(raw):
    """raw label probabilities into a distribution over option ids."""
    mass = sum(raw.values())
    if mass <= 0:
        return dict.fromkeys(raw, 0.0), 0.0
    return {k: v / mass for k, v in raw.items()}, mass


def confidence(probs):
    return max(probs.values()) if probs else 0.0


def margin(probs):
    """top minus second. zero when there is nothing to come second."""
    if len(probs) < 2:
        return 0.0
    ordered = sorted(probs.values(), reverse=True)
    return ordered[0] - ordered[1]


def concentration(probs):
    """normalised entropy: 0 spread evenly, 1 all on one option.

    this is what the jev protocol means by `confidence`, and it is a different
    number from the top probability. both are reported so that a client tuned
    against either is not silently mis-gated.
    """
    n = len(probs)
    if n < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(n)))


def expected_level(probs):
    """the probability weighted mean level index, for score questions."""
    return sum(int(level) * p for level, p in probs.items())


def answer(question, raw, option_mass_floor, option_mass=None, measured=True):
    """the full result object for one question.

    `option_mass` overrides the mass implied by `raw`. debiasing rescales the
    raw values, and the mass has to stay what the readout actually measured or
    it stops being a health check on the prompt.

    `measured` is false for a readout through a decision head, whose
    distribution covers the options and nothing else: its sum is 1 by
    construction, so the mass is reported as absent rather than as healthy.
    """
    probs, implied = renormalise(raw)
    mass = (implied if option_mass is None else option_mass) if measured else None
    flags = []
    if not measured:
        flags.append("no_option_mass")
    elif mass < option_mass_floor:
        flags.append("low_option_mass")
    top = max(probs, key=probs.get) if implied > 0 else None
    out = {
        "type": question.kind,
        "probs": probs,
        "top": top,
        "confidence": confidence(probs),
        "margin": margin(probs),
        "concentration": concentration(probs),
        "option_mass": mass,
        "calibrated": False,
        "flags": flags,
    }
    if question.kind == "score":
        out["score"] = expected_level(probs)
        out["legend"] = question.legend
    return out


class Agreement(typing.NamedTuple):
    """whether two readouts of one prompt are the same readout."""
    ok: bool
    why: str = ""


def agree(a, b, tolerance=None):
    """do two answers to the same prompt agree, allowing for the stack?

    spec/SPEC.md 12.0: prompts are exact and probabilities are not. the server
    packs prompts into batches whose composition depends on concurrent work,
    and matmul reduction order follows shape, so the same prompt read twice can
    differ by up to the measured drift.

    demanding equality reports a difference that means nothing. demanding
    nothing reports agreement that means nothing. so:

    - probabilities and option mass within `probability_abs`
    - the CHOICE must match when the margin is wide enough to be stable
    - inside that margin the top two must match as a set, order unasserted,
      because a near-tie genuinely flips: 0.487 against 0.432, measured
    """
    limits = tolerance or spec.constants()["numeric_tolerance"]
    close = limits["probability_abs"]

    if (a["option_mass"] is None) != (b["option_mass"] is None):
        return Agreement(False, "one readout measured option mass and the other is a head")
    if a["option_mass"] is not None and abs(a["option_mass"] - b["option_mass"]) > close:
        return Agreement(False, f"option mass {a['option_mass']:.4g} vs "
                                f"{b['option_mass']:.4g}")

    for key in set(a["probs"]) | set(b["probs"]):
        pa, pb = a["probs"].get(key, 0.0), b["probs"].get(key, 0.0)
        if abs(pa - pb) > close:
            return Agreement(False, f"probability for {key!r}: "
                                    f"{pa:.4g} vs {pb:.4g}")

    if a["top"] != b["top"]:
        if min(margin(a["probs"]), margin(b["probs"])) >= limits["stable_choice_margin"]:
            return Agreement(False, f"choice {a['top']!r} vs {b['top']!r} "
                                    f"on a margin that should be stable")
        if top_two(a["probs"]) != top_two(b["probs"]):
            return Agreement(False, f"choice {a['top']!r} vs {b['top']!r} "
                                    f"and the top two differ")
    return Agreement(True)


def top_two(probs):
    return frozenset(sorted(probs, key=probs.get, reverse=True)[:2])
