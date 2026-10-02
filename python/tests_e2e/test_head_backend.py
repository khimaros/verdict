"""the jev endpoint in front of a model that answers through a decision head.

the backend is `../fake-openai --llamacpp --pooling <none|last>`: llama.cpp
started as an embedding server, answering the hidden state the test scripted
at the last position of whatever prompt arrives, and recording what it was
asked. so the suite sees the prompt as it reached the backend and holds the
answer to the head's own arithmetic, with no weights anywhere.

the head is a toy whose rows each read one dimension, so the hidden state
`[5, 0, 0, 0]` makes the first option win and `[0, 5, 0, 0]` the second. the
first is the mock's default, which is what verdict's startup verification
reads; a test that wants another answer queues it.

skips cleanly when the mock is not built. `make` in ../fake-openai builds it.
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python" / "tests"))
sys.path.insert(0, str(ROOT / "python" / "tests_e2e"))

import test_fake_backend as mocked  # noqa: E402
from test_fake_backend import CRITERIA, TICKET, ask  # noqa: E402
from test_head import authors_prompt, write_head  # noqa: E402

pytestmark = mocked.pytestmark

WIDTH = 4
# as many rows as the verification's longest list, each reading one dimension
ROWS = [[float(i % WIDTH == d) for d in range(WIDTH)] for i in range(52)]
FIRST = [5.0, 0.0, 0.0, 0.0]
SECOND = [0.0, 5.0, 0.0, 0.0]
QUEUE = "Which queue should handle this?"
QUESTIONS = {"queue": {"type": "choice", "instructions": QUEUE, "criteria": CRITERIA}}
PROMPT = authors_prompt(TICKET, QUEUE, [f"{name}: {text}" for name, text in CRITERIA.items()])
OAI, NATIVE = "/v1/embeddings", "/embedding"


def head_file(tmp_path_factory):
    return write_head(tmp_path_factory.mktemp("head") / "head.npz", weight=ROWS,
                      bias=[0.0] * len(ROWS), mu=[0.0] * WIDTH, sd=[1.0] * WIDTH)


def serving(tmp_path_factory, *args, pooling="last", model=mocked.MODEL, spec=None):
    """verdict in front of the mock as an embedding server whose every
    unscripted read answers the first option."""
    return mocked.serving(
        tmp_path_factory, *args, model=model, mock=("--pooling", pooling), spec=spec,
        prepare=lambda fake: fake.set_default_embedding({"vector": FIRST}))


def asked(fake, route):
    """the bodies the backend received on one route, oldest first."""
    return [c["body"] for c in fake.captures() if c["path"].endswith(route)]


@pytest.fixture(scope="module")
def endpoint(tmp_path_factory):
    args = ("--layout", "jev-omni", "--head", head_file(tmp_path_factory))
    with serving(tmp_path_factory, *args) as (base, fake, log, server):
        assert server.poll() is None, log.read_text()
        yield base, fake, log


def test_the_answer_is_the_head_applied_to_the_last_position(endpoint):
    base, fake, _log = endpoint
    fake.program_embeddings({"vector": SECOND})
    status, wire = ask(base, questions=QUESTIONS)
    assert status == 200, wire
    top = math.exp(5) / (math.exp(5) + 2)
    assert wire["answers"]["queue"]["choice"] == "access"
    assert wire["answers"]["queue"]["probabilities"] == pytest.approx(
        {"billing": (1 - top) / 2, "access": top, "technical": (1 - top) / 2}, abs=1e-4)


def test_the_backend_is_sent_the_authors_prompt_and_asked_for_no_logits(endpoint):
    base, fake, _log = endpoint
    ask(base, questions=QUESTIONS)
    body = asked(fake, OAI)[-1]
    assert body["input"] == PROMPT
    # a pooled vector is normalised unless asked otherwise, and the head was
    # trained on the raw hidden state
    assert body["embd_normalize"] == -1
    assert not asked(fake, "/completion")


def test_the_input_tokens_are_the_servers_own_count(endpoint):
    """the route that returns one vector also says how long the prompt was,
    so the count costs no round trip of its own."""
    base, _fake, _log = endpoint
    status, wire = ask(base, questions=QUESTIONS)
    assert status == 200, wire
    # the mock's vocabulary is one token per byte
    assert wire["usage"]["input_tokens"] == len(PROMPT.encode())


def test_the_banner_says_the_model_is_read_through_a_head(endpoint):
    _base, _fake, log = endpoint
    banner = log.read_text()
    assert "layout    jev-omni" in banner
    assert "head" in banner and "3/3" in banner


def test_a_server_returning_every_position_is_read_and_counted_by_its_rows(tmp_path_factory):
    args = ("--layout", "jev-omni", "--head", head_file(tmp_path_factory))
    with serving(tmp_path_factory, *args, pooling="none") as (base, fake, log, server):
        assert server.poll() is None, log.read_text()
        fake.program_embeddings({"vector": SECOND})
        status, wire = ask(base, questions=QUESTIONS)
        assert status == 200, wire
        assert wire["answers"]["queue"]["choice"] == "access"
        assert wire["usage"]["input_tokens"] == len(PROMPT.encode())
        # the refusal is learned once, not paid on every decision
        assert len(asked(fake, OAI)) == 1 and len(asked(fake, NATIVE)) > 1
        assert asked(fake, NATIVE)[-1]["content"] == PROMPT


def test_a_head_model_nothing_can_apply_is_still_refused(tmp_path_factory):
    with serving(tmp_path_factory, model=mocked.UNREADABLE,
                 spec=mocked.spec_with(tmp_path_factory, mocked.HEADED)) as (
            _b, _f, log, server):
        assert server.wait(timeout=30) != 0
        assert "--layout" in log.read_text()


def test_a_recognised_head_model_needs_no_layout_flag(tmp_path_factory):
    args = ("--head", head_file(tmp_path_factory))
    with serving(tmp_path_factory, *args, model="jev-omni:Q8_0") as (
            base, _fake, log, server):
        assert server.poll() is None, log.read_text()
        assert "layout    jev-omni" in log.read_text()
        assert ask(base, questions=QUESTIONS)[0] == 200
