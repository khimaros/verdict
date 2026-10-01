"""http backend: a llama-server, optionally behind llama-swap.

stdlib only. the rust core will replace this for the runtime path; this
implementation is the reference and the evaluation harness, so it stays thin.
"""

import http.client
import io
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse

from . import spec

HTTP_TIMEOUT = 900
# llama-swap answers 502 and llama-server 503 while a model is still loading.
# that is a wait rather than a failure, and a jev client that never retries
# (jevbench) would otherwise lose the first decision against every cold model.
LOADING_STATUSES = (502, 503)
LOADING_BACKOFF_S = (0.5, 1, 2, 4, 8)
# the huggingface cache layout: .../models--<org>--<name>/snapshots/<rev>/<file>
HUB_CACHE_PATH = re.compile(r"/models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]+)/(.+)$")


def weights_from_props(props):
    """the weights a server loaded, as repo, revision and file.

    the served model name is not a stable key, since a config reload can put
    another file behind it. the gguf path is, and the hub cache names the repo
    and revision in it. a path outside the cache is reported as it is.
    """
    path = props.get("model_path")
    if not path:
        return None
    m = HUB_CACHE_PATH.search(path)
    if not m:
        return {"repo": None, "revision": None, "file": path}
    org, name, revision, file = m.groups()
    return {"repo": f"{org}/{name}", "revision": revision, "file": file}


class HttpBackend:
    """llama-swap serves the native endpoints under /upstream/<model>/ and not
    at the root, so the path prefix is resolved once here rather than at every
    call site. a bare llama-server takes the same client with upstream unset.
    """

    def __init__(self, base_url, model=None, upstream=None, timeout=HTTP_TIMEOUT,
                 cache_prompt=True):
        base = base_url.rstrip("/").removesuffix("/v1")
        self.root = base
        # default to llama-swap routing when a model is named, since that is
        # the deployment where the root endpoints 404
        self.upstream = bool(model) if upstream is None else upstream
        self.base = f"{base}/upstream/{model}" if self.upstream and model else base
        self.model = model
        self.timeout = timeout
        # reusing a prefix is most of why a decision is cheap, and it is also
        # the suspect for probabilities that are not bit-identical run to run:
        # a partly reused prompt prefills in different batch shapes from a cold
        # one. off is the control, not a supported serving mode.
        self.cache_prompt = cache_prompt
        url = urllib.parse.urlsplit(base)
        self._connection_class = (http.client.HTTPSConnection if url.scheme == "https"
                                  else http.client.HTTPConnection)
        self._netloc = url.netloc
        # open connections kept for reuse: a fresh one costs a round trip before
        # the request is sent. the jev server answers every client connection
        # on a new thread, so connections are pooled across threads rather than
        # held per thread, and each carries one exchange at a time.
        self._idle = []
        self._idle_lock = threading.Lock()

    def close(self):
        with self._idle_lock:
            idle, self._idle = self._idle, []
        for conn in idle:
            conn.close()

    def _take(self):
        with self._idle_lock:
            if self._idle:
                return self._idle.pop()
        return self._connection_class(self._netloc, timeout=self.timeout)

    def _give_back(self, conn):
        with self._idle_lock:
            self._idle.append(conn)

    def _exchange(self, method, url, body):
        """one request on a pooled connection, retried once on a fresh one if
        the server closed the pooled one while it sat idle."""
        path = urllib.parse.urlsplit(url)
        target = path.path + (f"?{path.query}" if path.query else "")
        headers = {"content-type": "application/json"} if body is not None else {}
        for attempt in (1, 2):
            conn = self._take()
            try:
                conn.request(method, target, body=body, headers=headers)
                r = conn.getresponse()
                data = r.read()
            except (http.client.RemoteDisconnected, ConnectionError, http.client.CannotSendRequest):
                conn.close()
                if attempt == 2:
                    raise
                continue
            except BaseException:
                conn.close()
                raise
            self._give_back(conn)
            return r, data

    def _open(self, url, body=None):
        """the parsed reply, waiting out a model that is still loading.

        failures surface as urllib's HTTPError, which is what the retry here and
        the server's mapping of a refused prompt to 422 are written against.
        """
        for delay in (*LOADING_BACKOFF_S, None):
            r, data = self._exchange("POST" if body is not None else "GET", url, body)
            if r.status < 400:
                return json.loads(data)
            error = urllib.error.HTTPError(url, r.status, r.reason, r.headers, io.BytesIO(data))
            if r.status not in LOADING_STATUSES or delay is None:
                raise error
            time.sleep(delay)

    def _post(self, path, payload):
        t0 = time.monotonic()
        out = self._open(self.base + path, json.dumps(payload).encode())
        return out, (time.monotonic() - t0) * 1000.0

    def props(self):
        return self._open(self.base + "/props")

    def readout(self):
        """how the model registry says this model must be read, or None.

        a llama-swap config generated from the registry advertises it on
        /v1/models as `meta.llamaswap.readout`. a bare llama-server, or a
        model the registry says nothing about, answers None.
        """
        try:
            listing = self._open(self.root + "/v1/models")
        except (OSError, http.client.HTTPException, ValueError):
            return None
        for entry in listing.get("data", []):
            if entry.get("id") == self.model:
                return ((entry.get("meta") or {}).get("llamaswap") or {}).get("readout")
        return None

    def tokenize(self, text, add_special=False):
        """special tokens must be parsed, or the chat markers become literal text."""
        out, _ = self._post("/tokenize", {"content": text, "add_special": add_special,
                                          "parse_special": True})
        return out["tokens"]

    def score(self, prompt, label_ids, delimiters=None):
        """probabilities for the given label token ids at the next position.

        `prompt` is a token id list or the prompt text. text costs no
        `/tokenize` round trip and is byte-equivalent whenever the spec's
        boundary assertion holds, which is what makes it safe to prefer.

        `delimiters` are llama-server message delimiters: it checkpoints where
        a user one matches, which is what lets a sliding-window or recurrent
        model resume a shared prefix. a server without them ignores the field.

        widens the readout until every label is present. a label missing from a
        truncated candidate list is an artifact, not a zero, and renormalising
        over the survivors looks like a confident answer without being one.
        """
        ladder = spec.constants()["n_probs_ladder"]
        wall, retries = 0.0, 0
        by_id, resp, entries, n_probs = {}, None, [], ladder[0]
        extra = {"message_delimiters": delimiters} if delimiters else {}

        for n_probs in ladder:
            resp, ms = self._post("/completion", {
                "prompt": prompt, "n_predict": 1, "n_probs": n_probs,
                "cache_prompt": self.cache_prompt, "post_sampling_probs": False,
                "temperature": -1.0, **extra})
            wall += ms
            entries = resp["completion_probabilities"][0]["top_logprobs"]
            by_id = {e["id"]: math.exp(e["logprob"]) for e in entries}
            # the server returns min(n_probs, vocab), so a list shorter than
            # asked for proves we saw the whole distribution and that a still
            # missing label is genuinely absent rather than cut off
            if all(i in by_id for i in label_ids) or len(entries) < n_probs:
                truncated = False
                break
            retries += 1
        else:
            # the ladder ran out with a label still unaccounted for
            truncated = True

        return {
            "raw": {i: by_id.get(i, 0.0) for i in label_ids},
            "truncated": truncated,
            "n_probs_used": n_probs,
            "retries": retries,
            "score_ms": wall,
            "prompt_n": resp["timings"]["prompt_n"],
            # evaluated plus reused: the real prompt length, which the client
            # does not otherwise know when it sends text instead of token ids
            "prompt_total": (resp["timings"]["prompt_n"]
                             + resp["timings"].get("cache_n", 0)),
            "tokens_cached": resp.get("tokens_cached"),
            "top_unconstrained": entries[0]["token"] if entries else None,
        }
