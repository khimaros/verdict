"""the backend keeps one connection open rather than paying a handshake per call.

every decision is at least one http call to llama-server, and a fresh tcp
connection costs a full round trip before the request is even sent. over the
~150 ms link to the eval server that was half of a small model's decision.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llama_verdict.backend import HttpBackend


class KeepAlive(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    connections = 0

    def setup(self):
        super().setup()
        type(self).connections += 1

    def _reply(self, body):
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._reply({"chat_template": "", "model_path": "/m.gguf"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
        self._reply({"tokens": [1, 2]})

    def log_message(self, *args):
        pass


@pytest.fixture
def upstream():
    KeepAlive.connections = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), KeepAlive)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", KeepAlive
    server.shutdown()


def test_several_calls_share_one_connection(upstream):
    url, handler = upstream
    backend = HttpBackend(url, upstream=False)
    for _ in range(3):
        backend.tokenize("hello")
    backend.props()
    assert handler.connections == 1


def test_calls_from_successive_threads_share_one_connection(upstream):
    # the jev server answers each client connection on a thread of its own,
    # and clients open one per request, so a per-thread connection would be
    # opened afresh for every decision
    url, handler = upstream
    backend = HttpBackend(url, upstream=False)
    for _ in range(3):
        t = threading.Thread(target=backend.tokenize, args=("hello",))
        t.start()
        t.join()
    assert handler.connections == 1


def test_concurrent_calls_each_get_a_connection(upstream):
    url, _handler = upstream
    backend = HttpBackend(url, upstream=False)
    errors = []

    def call():
        try:
            backend.tokenize("hello")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_a_dropped_connection_is_reopened(upstream):
    url, handler = upstream
    backend = HttpBackend(url, upstream=False)
    backend.tokenize("hello")
    backend.close()
    backend.tokenize("again")
    assert handler.connections == 2
