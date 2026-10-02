"""clef's joint schema head, ported to numpy, against the author's own class.

the head is a small transformer over every position's hidden state and its
only published implementation is torch. a port that is slightly wrong still
returns a confident distribution per question, so it is held to logits the
author's `JointSchemaHead` computed: `scripts/make_joint_head_fixture.py`
builds that class at toy size with seeded weights and records what it returned
(Cloudflare/clef-flash @ 17f0b0a, joint_schema_model.py).
"""

import http.server
import json
import os
import threading

import pytest

np = pytest.importorskip("numpy")

from llama_verdict import head as head_files  # noqa: E402
from llama_verdict import joint_head, joint_prompt  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "joint_head.json")
# torch computed these in float32
TOLERANCE = 1e-4


def array(flat):
    return np.asarray(flat["values"], dtype=np.float32).reshape(flat["shape"])


@pytest.fixture(scope="module")
def golden():
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def head(golden):
    return joint_head.JointSchema({name: array(t) for name, t in golden["tensors"].items()},
                                  golden["config"]["heads"])


def questions(golden):
    return [joint_prompt.Spans(q["type"], tuple(q["question_span"]),
                               tuple(tuple(o) for o in q["option_spans"]))
            for q in golden["questions"]]


def logits(head, golden, asked=None):
    embedding = array(golden["embedding"])
    return head.logits(array(golden["hidden"]), golden["input_ids"],
                       questions(golden) if asked is None else asked,
                       lambda ids: embedding[ids])


def test_the_port_returns_the_authors_logits(head, golden):
    for ours, theirs in zip(logits(head, golden), golden["logits"], strict=True):
        assert ours.tolist() == pytest.approx(theirs, abs=TOLERANCE)


def test_every_question_is_decided_jointly(head, golden):
    """the fields attend to each other, so a question asked alone is a
    different computation from the same question asked beside another. a port
    that scored them independently would pass a single-question check."""
    both = logits(head, golden)[0]
    alone = logits(head, golden, questions(golden)[:1])[0]
    assert np.abs(both - alone).max() > TOLERANCE


def test_probabilities_are_a_softmax_per_question(head, golden):
    for row, theirs in zip(head.probabilities(logits(head, golden)), golden["logits"],
                           strict=True):
        e = np.exp(np.asarray(theirs) - max(theirs))
        assert row == pytest.approx((e / e.sum()).tolist(), abs=TOLERANCE)
        assert sum(row) == pytest.approx(1.0)


def test_the_head_is_read_from_safetensors(tmp_path, golden):
    """the release ships the head as safetensors: an 8 byte length, a json
    header of dtype, shape and byte range per tensor, then the raw values."""
    tensors = {name: array(t) for name, t in golden["tensors"].items()}
    header, blob = {}, b""
    for name, value in tensors.items():
        raw = value.astype("<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(value.shape),
                        "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    encoded = json.dumps(header).encode()
    path = tmp_path / "joint_head.safetensors"
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + blob)
    loaded = joint_head.read_safetensors(str(path))
    assert sorted(loaded) == sorted(tensors)
    for name, value in tensors.items():
        assert np.array_equal(loaded[name], value)


class Ranges(http.server.BaseHTTPRequestHandler):
    """a file host that honours byte ranges, as the hub does."""
    blob = b""
    asked = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        start, end = (int(n) for n in self.headers["Range"].removeprefix("bytes=").split("-"))
        self.asked.append((start, end))
        body = self.blob[start:end + 1]
        self.send_response(206)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def bf16_file(tensors):
    """a safetensors file of bf16 tensors, and the float32 values it holds."""
    header, blob, held = {}, b"", {}
    for name, values in tensors.items():
        halves = (values.astype("<f4").view("<u4") >> 16).astype("<u2")
        held[name] = (halves.astype("<u4") << 16).view("<f4").reshape(values.shape)
        header[name] = {"dtype": "BF16", "shape": list(values.shape),
                        "data_offsets": [len(blob), len(blob) + halves.nbytes]}
        blob += halves.tobytes()
    encoded = json.dumps(header).encode()
    return len(encoded).to_bytes(8, "little") + encoded + blob, held


@pytest.fixture
def hub(monkeypatch):
    rng = np.random.default_rng(7)
    blob, held = bf16_file({"other.weight": rng.normal(size=(3, 8)),
                            "lm_head.weight": rng.normal(size=(20, 8))})
    Ranges.blob, Ranges.asked = blob, []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Ranges)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(head_files, "HUB_URL", f"http://127.0.0.1:{server.server_port}/{{file}}")
    yield held["lm_head.weight"]
    server.shutdown()


ROWS = {"repo": "someone/model", "revision": "a" * 40, "file": "model-00001.safetensors",
        "tensor": "lm_head.weight"}


def test_embedding_rows_are_read_by_range_and_not_the_whole_file(tmp_path, hub):
    rows = joint_head.Rows(ROWS, str(tmp_path))
    assert np.array_equal(rows([3, 17, 4, 3]), hub[[3, 17, 4, 3]])
    fetched = sum(end - start + 1 for start, end in Ranges.asked)
    assert fetched < len(Ranges.blob) / 2
    # adjacent rows come in one request: the length, the header, 3-4 and 17
    assert len(Ranges.asked) == 4


def test_a_row_fetched_once_is_kept(tmp_path, hub):
    joint_head.Rows(ROWS, str(tmp_path))([3, 4])
    before = len(Ranges.asked)
    again = joint_head.Rows(ROWS, str(tmp_path))
    assert np.array_equal(again([4, 3]), hub[[4, 3]])
    assert len(Ranges.asked) == before


def test_bf16_tensors_are_widened_exactly(tmp_path):
    """a bf16 value is the top half of a float32, so widening it is exact."""
    values = np.asarray([1.0, -2.5, 0.15625, 1024.0], dtype=np.float32)
    halves = (values.view("<u4") >> 16).astype("<u2").tobytes()
    header = json.dumps({"w": {"dtype": "BF16", "shape": [4],
                               "data_offsets": [0, len(halves)]}}).encode()
    path = tmp_path / "bf16.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + halves)
    assert joint_head.read_safetensors(str(path))["w"].tolist() == values.tolist()
