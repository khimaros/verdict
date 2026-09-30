"""conformance against spec/fixtures.

every implementation runs the equivalent of this file. a failure here means the
implementation and the spec disagree, and the spec wins.
"""

import glob
import json
import os

import pytest

from llama_verdict import prompt, spec, types

SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spec")
FIXTURES = sorted(glob.glob(os.path.join(SPEC, "fixtures", "*.json")))
# the same cases rendered under each named layout in spec/layouts
LAYOUT_FIXTURES = sorted(glob.glob(os.path.join(SPEC, "fixtures", "layouts", "*", "*.json")))
ALL_FIXTURES = FIXTURES + LAYOUT_FIXTURES


def load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def reference():
    return types.Formatter.load(os.path.join(SPEC, "formatters", "reference.json"))


def formatter_for(fixture, reference):
    """the reference formatter, or a named layout's."""
    name = fixture["formatter"]
    if name == "reference":
        return reference
    return types.Formatter.for_layout(name.removeprefix("layout:"), reference)


def ids(paths):
    return [os.path.relpath(p, os.path.join(SPEC, "fixtures"))[: -len(".json")]
            for p in paths]


def test_fixtures_exist():
    assert len(FIXTURES) >= 20, "the spec requires at least 20 conformance fixtures"


def test_every_layout_has_fixtures():
    for name in os.listdir(os.path.join(SPEC, "layouts")):
        layout = name.removesuffix(".json")
        assert any(f"/layouts/{layout}/" in p for p in LAYOUT_FIXTURES), layout


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=ids(ALL_FIXTURES))
def test_prefix_matches(path, reference):
    f = load(path)
    formatter = formatter_for(f, reference)
    assert prompt.render_prefix(formatter, f["state"]) == f["expected"]["prefix"]


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=ids(ALL_FIXTURES))
def test_suffixes_match(path, reference):
    f = load(path)
    formatter = formatter_for(f, reference)
    questions = types.parse_questions(f["questions"], formatter.layout)
    assert [q.name for q in questions] == list(f["expected"]["questions"])
    for q in questions:
        want = f["expected"]["questions"][q.name]
        assert q.kind == want["kind"]
        assert prompt.render_suffix(formatter, q) == want["suffix"]
        assert prompt.label_map(q, layout=formatter.layout) == want["labels"]
        assert prompt.alphabet_flags(q) == want["flags"]
        if "legend" in want:
            assert q.legend == want["legend"]


@pytest.mark.parametrize("path", FIXTURES, ids=ids(FIXTURES))
def test_labels_are_declaration_order(path, reference):
    """option order reaches the prompt, so it is part of the contract."""
    f = load(path)
    for q in types.parse_questions(f["questions"]):
        labels = spec.labels(len(q.options))
        assert list(prompt.label_map(q).values()) == labels


def test_key_order_does_not_change_the_prompt(reference):
    """the state is the cached prefix; an unstable serialisation would destroy
    prefix reuse while still returning correct answers."""
    a = load(os.path.join(SPEC, "fixtures", "state-object.json"))
    b = load(os.path.join(SPEC, "fixtures", "state-object-key-order.json"))
    assert a["state"] != b["state"] or list(a["state"]) != list(b["state"])
    assert (prompt.render_prefix(reference, a["state"])
            == prompt.render_prefix(reference, b["state"]))


def test_null_and_empty_description_fall_back_to_the_option_id():
    for fixture in ("choice-null-description", "choice-empty-description"):
        f = load(os.path.join(SPEC, "fixtures", fixture + ".json"))
        q = types.parse_questions(f["questions"])[0]
        assert q.options[0].description == q.options[0].id


def test_noul_is_normalised_to_boolean():
    f = load(os.path.join(SPEC, "fixtures", "boolean-wire-alias-noul.json"))
    assert types.parse_questions(f["questions"])[0].kind == types.BOOLEAN


def test_boolean_puts_true_first():
    """P(true) is read off the first option, so the order is load bearing."""
    q = types.parse_question("b", {"type": "boolean", "instructions": "Is it?"})
    assert [o.id for o in q.options] == ["true", "false"]


def test_label_alphabet_is_letters_only():
    labels = spec.labels(52)
    assert labels[:3] == ["A", "B", "C"]
    assert labels[26] == "a"
    assert labels[-1] == "z"
    assert all(ell.isalpha() for ell in labels)


def test_too_many_options_is_refused_with_a_usable_message():
    with pytest.raises(ValueError) as e:
        spec.labels(53)
    assert "tournament" in str(e.value), "the error must name the supported alternative"


def test_a_single_option_is_legal_and_flagged():
    """browser-use's jev agent sends a target question for an operation that
    only one element supports. rejecting it breaks a real client."""
    q = types.parse_question("q", {"type": "choice", "instructions": "?",
                                   "criteria": {"only": "one"}})
    assert [o.id for o in q.options] == ["only"]
    assert "single_option" in prompt.alphabet_flags(q)


def test_zero_options_is_refused():
    with pytest.raises(ValueError):
        types.parse_question("q", {"type": "choice", "instructions": "?",
                                   "criteria": {}})


def test_synthetic_formatter_refuses_to_score(reference):
    """the reference formatter renders fixtures. it is not a chat format."""
    with pytest.raises(ValueError) as e:
        reference.check_usable()
    assert "synthetic" in str(e.value)


PINNED = sorted(p for p in glob.glob(os.path.join(SPEC, "formatters", "*.json"))
                if not p.endswith("reference.json"))


@pytest.mark.parametrize("path", PINNED, ids=ids(PINNED))
def test_pinned_formatters_are_usable(path):
    """a pinned formatter carries the measurement that justifies trusting it."""
    f = types.Formatter.load(path)
    f.check_usable()
    assert f.template_sha256, "a pinned formatter must record the template it came from"
    assert f.verification["split_clean"] is True
    assert f.verification["labels_single_token"] is True
    assert f.assistant_open, "the assistant opening is what positions the label"


def test_a_formatter_without_verification_is_refused():
    f = types.Formatter(model="untested", system_open="", system_close="",
                        user_open="", user_close="", assistant_open="X")
    with pytest.raises(ValueError) as e:
        f.check_usable()
    assert "verification" in str(e.value)


def test_a_formatter_below_the_mass_floor_is_refused():
    """a wrong opening still produces plausible answers, so this must be an
    error rather than a warning."""
    f = types.Formatter(model="broken", system_open="", system_close="", user_open="",
                        user_close="", assistant_open="X",
                        verification={"mean_option_mass": 1.7e-08})
    with pytest.raises(ValueError) as e:
        f.check_usable()
    assert "option mass" in str(e.value)


def test_a_layout_sets_its_own_mass_floor():
    """decider is trained on the letter rows only, so the rest of the
    vocabulary keeps a thin tail: measured 0.87-0.93 on the listed letters
    with the answer on top. the same mass from a chat model means the opening
    is wrong."""
    affixes = dict(system_open="", system_close="", user_open="", user_close="",
                   assistant_open="X", verification={"mean_option_mass": 0.87})
    types.Formatter(model="decider", layout="decider-plain", **affixes).check_usable()
    with pytest.raises(ValueError):
        types.Formatter(model="chat", **affixes).check_usable()
