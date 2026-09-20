"""what the serving path actually puts on the wire.

this exists because the question "does verdict send text or token ids?" was
answered wrongly twice in one session. the first answer came from calling
`HttpBackend.score` with a string and observing a string arrive, which only
proves the transport forwards what it is handed. the real path is
server -> jev -> Decider -> backend, and only exercising all of it answers
anything.

the prompt goes as TEXT. the spec requires
`tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`, so where
that holds the two forms are the same token sequence and text costs no
`/tokenize` round trip. text is also what makes a request readable in a proxy
log, which is where these get debugged.
"""

import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llama_verdict import server as vserver

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FORMATTER = os.path.join(REPO, "spec", "formatters", "qwen3.5-9b_Q8_0.json")

REQUEST = {"model": "jev-latest", "state": {"page": "front"},
           "questions": {"operation": {
               "type": "choice",
               "criteria": {"CLICK": "click it", "BACK": "go back"},
               "instructions": {"goal": "read comments"}}}}


class Upstream(BaseHTTPRequestHandler):
    """a llama-server that records what it was asked rather than answering it."""

    received = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        Upstream.received.append((self.path, body))
        if self.path.endswith("/tokenize"):
            payload = {"tokens": [9999]}
        else:
            payload = {"completion_probabilities": [{"top_logprobs": [
                           {"id": 32, "token": "A", "logprob": -0.1},
                           {"id": 33, "token": "B", "logprob": -2.0}]}],
                       "timings": {"prompt_n": 7, "cache_n": 0}}
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def served():
    """the real server, in front of a fake llama-server, on real sockets.

    one per module: the port is fixed, so a fixture per test would race its own
    predecessor for the bind.
    """
    upstream = HTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    port = 8481
    threading.Thread(target=vserver.main, args=([
        "--base-url", f"http://127.0.0.1:{upstream.server_address[1]}",
        "--model", "qwen3.5-9b:Q8_0", "--formatter", FORMATTER,
        "--port", str(port), "--quiet"],), daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    else:
        pytest.fail("the verdict server never came up")
    yield port
    upstream.shutdown()


@pytest.fixture
def wire(served):
    """a clean recording for each test, against the shared server."""
    Upstream.received = []
    return served


def ask(port, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/systemone", data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def test_the_prompt_goes_as_text(wire):
    ask(wire, REQUEST)
    completions = [b for path, b in Upstream.received if path.endswith("/completion")]
    assert completions, "the serving path made no /completion call"
    for body in completions:
        assert isinstance(body["prompt"], str), (
            f"prompt went as {type(body['prompt']).__name__}; a token array is "
            "unreadable in a proxy log and costs a /tokenize round trip")


def test_the_serving_path_makes_no_tokenize_round_trip(wire):
    """26 of 31 round trips were once one `/tokenize` per label letter. the
    label ids live in the formatter now and the prompt goes as text, so a
    scoring pass should be exactly one call."""
    ask(wire, REQUEST)
    paths = [path for path, _ in Upstream.received]
    assert not [p for p in paths if p.endswith("/tokenize")], paths


def test_cache_prompt_is_requested(wire):
    """the state is a shared prefix across the questions of one request and
    across the steps of an agent run; without this every one is a full
    prefill."""
    ask(wire, REQUEST)
    for path, body in Upstream.received:
        if path.endswith("/completion"):
            assert body["cache_prompt"] is True
