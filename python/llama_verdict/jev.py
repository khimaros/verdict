"""translation between verdict's result objects and the jev wire protocol.

docs/JEV_API.md records the protocol and where it disagrees with our spec. the
two disagreements that matter are handled here:

- jev's `confidence` is distribution concentration, not the top probability
- jev names the boolean type `noul`
"""

# jev clients validate that probabilities are numeric, in 0-1, and sum near 1.
# rounding is applied last so the sum survives it.
PRECISION = 4


def _round_distribution(probs):
    """round, then put any rounding residue on the largest entry.

    a client that checks the sum is near 1.0 should not be tripped by our own
    formatting, and browser-use/jev-ultrafast does check.
    """
    if not probs:
        return {}
    rounded = {k: round(v, PRECISION) for k, v in probs.items()}
    residue = round(1.0 - sum(rounded.values()), PRECISION)
    if residue:
        top = max(rounded, key=rounded.get)
        rounded[top] = round(rounded[top] + residue, PRECISION)
    return rounded


def answer_to_wire(answer):
    """one verdict answer in jev's shape."""
    kind = answer["type"]
    probs = _round_distribution(answer["probs"])

    if kind == "boolean":
        # jev reports only the probability of yes
        return {"type": "noul", "noul": probs.get("true", 0.0)}

    out = {"type": kind, "probabilities": probs,
           # jev means concentration by this word; our own `confidence` is the
           # top probability and is a different number
           "confidence": round(answer["concentration"], PRECISION)}
    if kind == "score":
        out["score"] = round(answer["score"], 2)
        out["legend"] = answer["legend"]
    else:
        out["choice"] = answer["top"]
    return out


def result_to_wire(result, model):
    """a whole decide() result as a /v1/systemone response body."""
    return {
        "model": model,
        "answers": {name: answer_to_wire(a) for name, a in result["answers"].items()},
        # nothing is sampled, so no tokens are ever produced. jev's own example
        # reports a nonzero count here and we cannot reproduce whatever it means.
        "usage": {"input_tokens": result["usage"]["input_tokens"], "output_tokens": 0},
    }


def validate_request(body):
    """the checks that produce a 422 rather than a traceback."""
    if not isinstance(body, dict):
        raise ValueError("request body must be a json object")
    for field in ("state", "questions"):
        if field not in body:
            raise ValueError(f"missing required field {field!r}")
    if not isinstance(body["questions"], dict):
        raise ValueError("`questions` must be a map of name to question")
    if not body["questions"]:
        raise ValueError("`questions` must not be empty")
    return body["state"], body["questions"]
