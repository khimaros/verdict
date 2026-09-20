"""comparing two readouts of the same prompt.

spec/SPEC.md 12.0: prompts are exact and probabilities are not. the same model
given the same prompt returns different probabilities run to run, because the
server packs prompts into batches whose composition depends on what else it is
doing, and matmul reduction order follows shape.

so a comparison that demands equality reports a difference that means nothing,
and one that demands nothing reports agreement that means nothing either.
"""

import pytest

from llama_verdict import extract, spec


def answer(probs, mass=1.0):
    top = max(probs, key=probs.get)
    return {"probs": probs, "top": top, "option_mass": mass,
            "confidence": probs[top]}


def test_identical_readouts_agree():
    a = answer({"x": 0.9, "y": 0.1})
    assert extract.agree(a, a).ok


def test_a_difference_inside_the_tolerance_agrees():
    """2e-3 was measured on a quiet server for the same prompt."""
    a = answer({"x": 0.947871, "y": 0.052129})
    b = answer({"x": 0.945722, "y": 0.054278})
    assert extract.agree(a, b).ok


def test_a_difference_beyond_the_tolerance_does_not():
    a = answer({"x": 0.90, "y": 0.10})
    b = answer({"x": 0.70, "y": 0.30})
    result = extract.agree(a, b)
    assert not result.ok
    assert "probability" in result.why


def test_a_flipped_choice_on_a_wide_margin_does_not_agree():
    """a clear winner changing IS a real disagreement."""
    a = answer({"x": 0.95, "y": 0.05})
    b = answer({"y": 0.95, "x": 0.05})
    assert not extract.agree(a, b).ok


def test_a_flipped_choice_inside_the_margin_agrees():
    """measured: 0.487 against 0.432 flipped between two runs of one prompt.
    demanding argmax-exact there reports a difference the stack cannot avoid,
    so the top two must match as a SET and the order is not asserted."""
    a = answer({"x": 0.487, "y": 0.432, "z": 0.081})
    b = answer({"y": 0.470, "x": 0.449, "z": 0.081})
    result = extract.agree(a, b)
    assert result.ok, result.why


def test_a_flip_to_an_option_outside_the_top_two_does_not_agree():
    """a near-tie excuses the order of the top two, never a third option
    arriving from nowhere."""
    a = answer({"x": 0.46, "y": 0.44, "z": 0.10})
    b = answer({"z": 0.46, "x": 0.44, "y": 0.10})
    assert not extract.agree(a, b).ok


def test_option_mass_is_compared_too():
    """a readout that agrees on the renormalised answer while the raw mass
    collapsed is not the same readout."""
    a = answer({"x": 0.9, "y": 0.1}, mass=0.99)
    b = answer({"x": 0.9, "y": 0.1}, mass=0.40)
    result = extract.agree(a, b)
    assert not result.ok
    assert "mass" in result.why


@pytest.mark.parametrize("name", ["probability_abs", "stable_choice_margin",
                                  "observed_max_drift"])
def test_the_tolerances_are_in_the_spec(name):
    assert name in spec.constants()["numeric_tolerance"]
