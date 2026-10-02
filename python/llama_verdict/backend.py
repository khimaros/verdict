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

from . import config, spec

HTTP_TIMEOUT = 900
# llama-swap answers 502 and llama-server 503 while a model is still loading.
# that is a wait rather than a failure, and a jev client that never retries
# (jevbench) would otherwise lose the first decision against every cold model.
LOADING_STATUSES = (502, 503)
LOADING_BACKOFF_S = (0.5, 1, 2, 4, 8)
# llama-server's answer to a request it will not take as sent, such as a
# prompt longer than its context
BACKEND_REFUSED = 400
# what a server pooling `none` says when asked on the openai embeddings route
POOLING_REFUSAL = b"pooling"
# asks for the hidden state as it is; a pooled vector is normalised by default
RAW_HIDDEN = {"embd_normalize": -1}
# a raw final hidden state is nowhere near unit length, and a normalised one
# is exactly that, so a norm this close to 1 is the server having normalised
UNIT_NORM_TOLERANCE = 1e-3
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
                 cache_prompt=True, api_key=None):
        base = base_url.rstrip("/").removesuffix("/v1")
        self.root = base
        # presented as a bearer on every request; the configured one for this
        # backend unless the caller names another
        self.api_key = api_key or config.backend_key(base_url)
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
        # whether the server answers one pooled vector on the openai embeddings
        # route; assumed until it refuses, see hidden()
        self._one_vector = True
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
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
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

    def registry_key(self):
        """the model registry's key for this model, or None.

        a llama-swap config generated from the registry names it on /v1/models
        as `meta.llamaswap.registry`. a bare llama-server, or an id the
        listing does not carry, answers None.
        """
        try:
            listing = self._open(self.root + "/v1/models")
        except (OSError, http.client.HTTPException, ValueError):
            return None
        for entry in listing.get("data", []):
            if entry.get("id") == self.model:
                return ((entry.get("meta") or {}).get("llamaswap") or {}).get("registry")
        return None

    def tokenize(self, text, add_special=False):
        """special tokens must be parsed, or the chat markers become literal text."""
        out, _ = self._post("/tokenize", {"content": text, "add_special": add_special,
                                          "parse_special": True})
        return out["tokens"]

    def hidden(self, prompt):
        """the unnormalised final hidden state at the last position of `prompt`,
        which is what a decision head reads. needs a llama-server in embedding
        mode.

        a server pooling `none` returns every position and one pooling `last`
        returns that one alone; the last row is the same vector either way.
        normalisation is declined in the request because a pooled vector is
        normalised by default and a head is trained on the raw one.

        the openai route is asked first: under pooling `last` it answers that
        one vector and the prompt's token count together. it refuses a server
        pooling `none`, which is then read through the native route for as long
        as this backend lives, the count being the rows that came back.
        """
        wall = 0.0
        if self._one_vector:
            try:
                out, wall = self._post("/v1/embeddings", {"input": prompt, **RAW_HIDDEN})
                vector, total = out["data"][0]["embedding"], out["usage"]["prompt_tokens"]
            except urllib.error.HTTPError as e:
                detail = e.read()
                if e.code != BACKEND_REFUSED or POOLING_REFUSAL not in detail.lower():
                    raise urllib.error.HTTPError(e.url, e.code, e.reason, e.headers,
                                                 io.BytesIO(detail)) from e
                self._one_vector = False
        if not self._one_vector:
            out, ms = self._post("/embedding", {"content": prompt, **RAW_HIDDEN})
            vector, total, wall = out[0]["embedding"][-1], len(out[0]["embedding"]), wall + ms
        if abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < UNIT_NORM_TOLERANCE:
            raise ValueError(
                f"{self.model}: the backend returned a normalised hidden state. a head "
                f"reads the raw one: start llama-server with --embd-normalize -1.")
        return {"hidden": vector, "score_ms": wall, "truncated": False, "retries": 0,
                "prompt_n": None, "tokens_cached": None, "prompt_total": total}

    def hidden_states(self, tokens):
        """the unnormalised final hidden state of EVERY position of a prompt
        given as token ids, which is what a joint head reads. needs a
        llama-server in embedding mode pooling `none`.

        the rows have to be the tokens sent, one for one: a head that scores an
        option by where it sits is wrong by a whole option if the server
        prepended a token or pooled the prompt away.
        """
        out, ms = self._post("/embedding", {"content": tokens, **RAW_HIDDEN})
        rows = out[0]["embedding"]
        if len(rows) != len(tokens):
            raise ValueError(
                f"{self.model}: sent {len(tokens)} tokens and got {len(rows)} hidden states "
                f"back. a joint head reads every position as sent: start llama-server with "
                f"--pooling none, on a model whose tokenizer adds no token of its own.")
        return {"hidden": rows, "score_ms": ms}

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
