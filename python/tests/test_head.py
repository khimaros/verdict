"""a model that answers through a trained decision head, read without weights.

such a model has no label tokens. its head is a matrix applied to the final
hidden state at the position verdict already reads, one row per option slot,
so the readout is the same single position with the author's rows in place of
the vocabulary's. these cases hold the arithmetic, the file format and the
prompt to the author's own code: akhilaaa3/Jev-Omni `jev_omni.py` and
ngquocvinh/Jev-Omni-GGUF `jev_omni_gguf_decide.py`.
"""

import array
import dataclasses
import hashlib
import math
import struct
import sys
import zipfile

import pytest

import llama_verdict
from llama_verdict import derive, extract, head, prompt, types
from llama_verdict.backend import HttpBackend
from llama_verdict.decide import Decider

LAYOUT = "jev-omni"
WEIGHT = [[1.0, 0.0, 0.5, 0.0], [0.0, 2.0, 0.0, 0.25], [0.5, 0.5, 0.5, 0.5]]
BIAS = [0.0, -0.5, 0.25]
MU = [0.5, 0.0, 1.0, 0.0]
SD = [2.0, 1.0, 0.5, 4.0]
HIDDEN = [1.5, 0.75, 2.0, -2.0]
MEETING = "The meeting starts at 10 AM. It is now 9 AM."


def npy(shape, values):
    """one float32 array in numpy's own file format, written without numpy."""
    header = f"{{'descr': '<f4', 'fortran_order': False, 'shape': {tuple(shape)!r}, }}"
    header += " " * (-(len(header) + 11) % 64) + "\n"
    return (b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode()
            + array.array("f", values).tobytes())


def write_head(path, weight=WEIGHT, bias=BIAS, mu=MU, sd=SD):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("linear.weight.npy", npy((len(weight), len(mu)), sum(weight, [])))
        z.writestr("linear.bias.npy", npy((len(bias),), bias))
        z.writestr("mu.npy", npy((1, len(mu)), mu))
        z.writestr("sd.npy", npy((1, len(sd)), sd))
    return str(path)


def authors_probabilities(count):
    """`decide()` in jev_omni_gguf_decide.py, over plain lists."""
    x = [(h - m) / s for h, m, s in zip(HIDDEN, MU, SD, strict=True)]
    z = [sum(w * v for w, v in zip(row, x, strict=True)) + b
         for row, b in zip(WEIGHT[:count], BIAS[:count], strict=True)]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    return [v / sum(e) for v in e]


def authors_prompt(state, question, options):
    """`_prompt()` in jev_omni.py inside the turn `decide()` wraps it in."""
    choices = "\n".join(f"{i + 1}. {value}" for i, value in enumerate(options))
    text = (f"{state}\n\n---\n\nQUESTION: {question}\n\nOPTIONS:\n{choices}\n\n"
            f"Reply with only the number of the correct option (1-{len(options)}).\n"
            "Output a single number and nothing else.")
    return f"<|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def rendered(state, name, body):
    formatter = types.Formatter.for_layout(LAYOUT, None)
    question = types.parse_question(name, body, LAYOUT)
    return prompt.render_prefix(formatter, state) + prompt.render_suffix(formatter, question)


@pytest.mark.parametrize("count", [2, 3])
def test_a_linear_head_is_the_authors_arithmetic(tmp_path, count):
    probs = head.load(write_head(tmp_path / "head.npz")).probs(HIDDEN, count)
    assert probs == pytest.approx(authors_probabilities(count), abs=1e-6)
    assert sum(probs) == pytest.approx(1.0)


def test_more_options_than_the_head_has_rows_is_refused(tmp_path):
    with pytest.raises(ValueError, match="3 option"):
        head.load(write_head(tmp_path / "head.npz")).probs(HIDDEN, 4)


def test_a_hidden_state_of_another_width_is_refused(tmp_path):
    """the wrong model behind the right head still yields a distribution."""
    with pytest.raises(ValueError, match="4"):
        head.load(write_head(tmp_path / "head.npz")).probs(HIDDEN[:3], 2)


def test_a_fetched_head_must_match_its_pinned_checksum(tmp_path, monkeypatch):
    source = write_head(tmp_path / "decision-head.npz")
    monkeypatch.setattr(head, "HUB_URL", f"file://{tmp_path}/{{file}}")
    declared = {"repo": "someone/some-gguf", "revision": "abc", "file": "decision-head.npz",
                "sha256": hashlib.sha256(open(source, "rb").read()).hexdigest()}
    fetched = head.fetch(declared, str(tmp_path / "cache"))
    assert head.load(fetched).probs(HIDDEN, 2) == pytest.approx(authors_probabilities(2), abs=1e-6)
    with pytest.raises(ValueError, match="sha256"):
        head.fetch(dict(declared, sha256="0" * 64), str(tmp_path / "other"))


def test_a_boolean_is_the_authors_yes_no_prompt():
    assert rendered(MEETING, "started", {
        "type": "noul", "instructions": "Has the meeting started?"}) == authors_prompt(
            MEETING, "Has the meeting started?", ["Yes", "No"])


def test_a_choice_numbers_its_options_and_says_how_many_there_are():
    marbles = "An urn holds one blue, one yellow, one red and one green marble."
    criteria = dict.fromkeys(["Blue", "Yellow", "Red", "Green"])
    assert rendered(marbles, "drawn", {
        "type": "choice", "instructions": "Which marble is drawn?",
        "criteria": criteria}) == authors_prompt(marbles, "Which marble is drawn?", list(criteria))


def test_the_layout_labels_as_many_options_as_the_head_has_slots():
    criteria = {f"o{i}": None for i in range(256)}
    question = types.parse_question("q", {"type": "choice", "instructions": "?",
                                           "criteria": criteria}, LAYOUT)
    labels = prompt.label_map(question, layout=LAYOUT)
    assert list(labels.values()) == [str(i + 1) for i in range(256)]
    assert prompt.alphabet_flags(question, LAYOUT) == []


def test_an_answer_read_through_a_head_reports_no_option_mass():
    """the head's softmax covers the options and nothing else, so its sum says
    nothing about whether the prompt steered the model. 1.0 would read as healthy."""
    question = types.parse_question("q", {"type": "noul", "instructions": "?"}, LAYOUT)
    a = extract.answer(question, {"true": 0.9, "false": 0.1}, 0.5, measured=False)
    assert a["option_mass"] is None
    assert a["flags"] == ["no_option_mass"]
    assert a["top"] == "true" and a["confidence"] == pytest.approx(0.9)


class HiddenBackend:
    """a backend that can only hand over a hidden state, as an embedding server does."""
    model = "jev-omni:Q8_0"

    def __init__(self):
        self.prompts = []

    def hidden(self, text):
        self.prompts.append(text)
        return {"hidden": HIDDEN, "score_ms": 1.0, "prompt_n": None, "prompt_total": 7,
                "tokens_cached": None, "retries": 0, "truncated": False}


def formatter_verified(correct, cases=3):
    return dataclasses.replace(
        types.Formatter.for_layout(LAYOUT, None),
        verification={"smoke_correct": f"{correct}/{cases}", "correct_ratio": correct / cases})


def test_a_decision_through_a_head_scores_the_hidden_state_and_no_logits(tmp_path):
    backend = HiddenBackend()
    decider = Decider(backend, formatter_verified(3),
                      head=head.load(write_head(tmp_path / "head.npz")))
    result = decider.decide(MEETING, {"started": {
        "type": "noul", "instructions": "Has the meeting started?"}})
    a = result["answers"]["started"]
    yes, no = authors_probabilities(2)
    assert a["probs"] == pytest.approx({"true": yes, "false": no}, abs=1e-6)
    assert a["option_mass"] is None and "no_option_mass" in a["flags"]
    assert backend.prompts == [authors_prompt(MEETING, "Has the meeting started?", ["Yes", "No"])]
    assert result["usage"]["input_tokens"] == 7


def test_a_normalised_hidden_state_is_refused(monkeypatch):
    """a unit vector points the same way as the raw one, so the head still
    answers, on numbers it was never trained on."""
    backend = HttpBackend("http://127.0.0.1:1", "jev-omni:Q8_0")
    monkeypatch.setattr(backend, "_post", lambda path, body: (
        {"data": [{"embedding": [0.6, 0.8]}], "usage": {"prompt_tokens": 9}}, 1.0))
    with pytest.raises(ValueError, match="normalis"):
        backend.hidden("a prompt")


def test_a_joint_head_without_numpy_says_what_to_install(monkeypatch, tmp_path):
    """numpy is optional, so its absence is a setup step to name and not a
    traceback: the endpoint reports a ValueError as one line."""
    # as if the module could not be imported: neither loaded nor loadable
    monkeypatch.setitem(sys.modules, "llama_verdict.joint_head", None)
    monkeypatch.delattr(llama_verdict, "joint_head", raising=False)
    with pytest.raises(ValueError, match="numpy"):
        head.for_layout("clef-flash", str(tmp_path / "joint_head.safetensors"))


def test_a_registry_readout_that_names_its_head_selects_that_layout():
    assert derive.layout_for("head:jev-omni") == LAYOUT


def test_a_head_readout_with_no_name_is_refused_for_not_saying_which():
    with pytest.raises(ValueError, match="--layout"):
        derive.layout_for("head")


def test_a_head_readout_naming_a_layout_read_by_logits_is_refused():
    """reading a head model's label logits measures its backbone."""
    with pytest.raises(ValueError, match="winnow"):
        derive.layout_for("head:winnow")


def test_a_head_formatter_that_failed_its_verification_is_refused():
    with pytest.raises(ValueError, match="1/3"):
        formatter_verified(1).check_usable()
