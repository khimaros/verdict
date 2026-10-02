"""the jev endpoint in front of a model read through a joint schema head.

the backend is `../fake-openai --llamacpp --pooling none`, which tokenizes one
token per byte and returns a hidden state per token id sent. the head is a toy
of clef's architecture whose weights are arranged so that its answer can be reasoned
about: the learned gate is shut, which leaves the lexical prior, and the
embedding rows are zero except for `b` and `l`. an option then scores by how
evenly its text mixes those two letters, so `blue` wins the verification and
`billing` wins the question asked here. what the suite holds is the plumbing a
unit test cannot see: token ids out, every position's state back, rows read
from a file, one request for all the questions.
"""

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python" / "tests_e2e"))

import test_fake_backend as mocked  # noqa: E402
from test_fake_backend import CRITERIA, TICKET, ask  # noqa: E402
from test_head_backend import NATIVE, asked  # noqa: E402

pytestmark = mocked.pytestmark

# the width of a hidden state the mock was not scripted for
WIDTH = 8
HEAD_WIDTH, FEEDFORWARD, VOCAB = 16, 16, 256
QUESTIONS = {
    "queue": {"type": "choice", "instructions": "Which queue should handle this?",
              "criteria": CRITERIA},
    "refund": {"type": "noul", "instructions": "Do they want money back?"},
}


def attention(prefix):
    return {f"{prefix}.in_proj_weight": (3 * HEAD_WIDTH, HEAD_WIDTH),
            f"{prefix}.in_proj_bias": (3 * HEAD_WIDTH,),
            f"{prefix}.out_proj.weight": (HEAD_WIDTH, HEAD_WIDTH),
            f"{prefix}.out_proj.bias": (HEAD_WIDTH,)}


def norm(prefix, size=HEAD_WIDTH):
    return {f"{prefix}.weight": (size,), f"{prefix}.bias": (size,)}


def shapes():
    """every tensor of a joint schema head with one routing and one decoder layer."""
    projections = ("memory", "question", "option_question", "global", "option_context",
                   "option_lexical")
    return {
        "prior_logit_scale": (), "joint_logit_scale": (), "residual_gate": (),
        **norm("hidden_norm", WIDTH),
        **{f"{name}_projection.weight": (HEAD_WIDTH, WIDTH) for name in projections},
        "type_embedding.weight": (3, HEAD_WIDTH),
        **norm("evidence_layers.0.query_norm"), **norm("evidence_layers.0.memory_norm"),
        **attention("evidence_layers.0.attention"), **norm("evidence_layers.0.feedforward_norm"),
        "evidence_layers.0.feedforward.0.weight": (FEEDFORWARD, HEAD_WIDTH),
        "evidence_layers.0.feedforward.0.bias": (FEEDFORWARD,),
        "evidence_layers.0.feedforward.3.weight": (HEAD_WIDTH, FEEDFORWARD),
        "evidence_layers.0.feedforward.3.bias": (HEAD_WIDTH,),
        **norm("option_summary_norm"),
        **attention("layers.0.self_attn"), **attention("layers.0.multihead_attn"),
        "layers.0.linear1.weight": (FEEDFORWARD, HEAD_WIDTH),
        "layers.0.linear1.bias": (FEEDFORWARD,),
        "layers.0.linear2.weight": (HEAD_WIDTH, FEEDFORWARD),
        "layers.0.linear2.bias": (HEAD_WIDTH,),
        **norm("layers.0.norm1"), **norm("layers.0.norm2"), **norm("layers.0.norm3"),
        **norm("field_norm"), **norm("option_norm"),
        "residual_scorer.0.weight": (HEAD_WIDTH, 4 * HEAD_WIDTH),
        "residual_scorer.0.bias": (HEAD_WIDTH,),
        "residual_scorer.3.weight": (1, HEAD_WIDTH), "residual_scorer.3.bias": (1,),
    }


def safetensors(path, tensors):
    header, blob = {}, b""
    for name, value in tensors.items():
        raw = np.asarray(value, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(np.shape(value)),
                        "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + blob)
    return str(path)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    rng = np.random.default_rng(11)
    tensors = {name: rng.normal(size=shape) for name, shape in shapes().items()}
    both = np.zeros(WIDTH)
    both[:2] = 1.0
    # the gate shut and the prior loud, on a hidden state that is one fixed
    # direction whatever the backend returns
    tensors.update({"residual_gate": -50.0, "prior_logit_scale": 9.0,
                    "hidden_norm.weight": np.zeros(WIDTH), "hidden_norm.bias": both})
    rows = np.zeros((VOCAB, WIDTH))
    rows[ord("b"), 0] = rows[ord("l"), 1] = 1.0
    folder = tmp_path_factory.mktemp("joint")
    return (safetensors(folder / "joint_head.safetensors", tensors),
            safetensors(folder / "rows.safetensors", {"lm_head.weight": rows}))


def serving(tmp_path_factory, files, pooling="none"):
    return mocked.serving(tmp_path_factory, "--layout", "clef-flash", "--head", files[0],
                          "--head-rows", files[1], mock=("--pooling", pooling))


@pytest.fixture(scope="module")
def endpoint(tmp_path_factory, files):
    with serving(tmp_path_factory, files) as (base, fake, log, server):
        assert server.poll() is None, log.read_text()
        yield base, fake, log


def test_every_question_is_answered_from_one_request(endpoint):
    base, fake, _log = endpoint
    before = len(asked(fake, NATIVE))
    status, wire = ask(base, questions=QUESTIONS)
    assert status == 200, wire
    assert wire["answers"]["queue"]["choice"] == "billing"
    assert wire["answers"]["queue"]["probabilities"]["billing"] > 0.99
    assert wire["answers"]["refund"]["type"] == "noul"
    assert len(asked(fake, NATIVE)) == before + 1


def test_the_prompt_goes_as_token_ids_and_holds_the_state_and_every_field(endpoint):
    base, fake, _log = endpoint
    ask(base, questions=QUESTIONS)
    sent = asked(fake, NATIVE)[-1]
    prompt = "".join(map(chr, sent["content"]))
    assert f"STATE:\n{TICKET}\n\nSCHEMA FIELDS:\n" in prompt
    assert "FIELD 1\nID: queue\nTYPE: choice" in prompt
    assert "FIELD 2\nID: refund\nTYPE: noul" in prompt
    assert prompt.endswith("JOINT SCHEMA DECISIONS:")
    assert sent["embd_normalize"] == -1


def test_the_usage_is_the_one_prompts_length(endpoint):
    base, fake, _log = endpoint
    _status, wire = ask(base, questions=QUESTIONS)
    assert wire["usage"]["input_tokens"] == len(asked(fake, NATIVE)[-1]["content"])


def test_the_banner_says_the_model_is_read_jointly(endpoint):
    _base, _fake, log = endpoint
    banner = log.read_text()
    assert "layout    clef-flash" in banner
    assert "3/3" in banner and "joint" in banner


def test_a_server_that_returns_one_vector_cannot_feed_a_joint_head(tmp_path_factory, files):
    """a joint head reads every position, which only pooling none returns."""
    with serving(tmp_path_factory, files, pooling="last") as (_base, _fake, log, server):
        assert server.wait(timeout=30) != 0
        assert "pooling none" in log.read_text()
