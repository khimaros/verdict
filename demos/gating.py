"""act only when the scorer says it knows.

the confidence to gate on is already computed and already in the response. the
jev client records it and never reads it, so a decision made at 0.27 is
executed exactly as eagerly as one made at 0.96.

measured on a hacker news run: the target confidence was 0.959 on the decision
that worked and 0.269, 0.378, 0.385 and 0.476 on the four that wandered, with
option mass at 0.999 throughout. the signal was there the whole time.

this converts a low-confidence decision into BLOCKED. for an autonomous demo,
escalation means stopping: it turns a wrong action into no action, which is a
real improvement in trustworthiness and is not the same thing as finishing the
task.
"""

import sys

# the head whose confidence decides whether an action happens at all
TARGET_SUFFIX = "_target"


def gate(result, threshold):
    """block the step when the operation's chosen target is not trusted."""
    answers = result.get("answers", {})
    operation = answers.get("operation", {})
    chosen = operation.get("choice")
    if not chosen:
        return result

    target = answers.get(chosen.lower() + TARGET_SUFFIX)
    if not target:
        return result

    probabilities = target.get("probabilities") or {}
    confidence = probabilities.get(target.get("choice"), 0.0)
    if confidence >= threshold:
        return result

    print(f"  gated: {chosen} target at {confidence:.3f} < {threshold:.2f}, "
          f"blocking instead of acting", file=sys.stderr)
    operation["choice"] = "BLOCKED"
    # the client validates that the reported distribution still covers the set
    # it declared and that the choice holds the maximum, so move the mass
    probs = operation.get("probabilities") or {}
    if "BLOCKED" in probs:
        for key in probs:
            probs[key] = 0.0
        probs["BLOCKED"] = 1.0
    return result
