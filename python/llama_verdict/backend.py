"""http backend: a llama-server, optionally behind llama-swap.

stdlib only. the rust core will replace this for the runtime path; this
implementation is the reference and the evaluation harness, so it stays thin.
"""

import json
import math
import time
import urllib.error
import urllib.request

from . import spec

HTTP_TIMEOUT = 900


class HttpBackend:
    """llama-swap serves the native endpoints under /upstream/<model>/ and not
    at the root, so the path prefix is resolved once here rather than at every
    call site. a bare llama-server takes the same client with upstream unset.
    """

    def __init__(self, base_url, model=None, upstream=None, timeout=HTTP_TIMEOUT,
                 cache_prompt=True):
        base = base_url.rstrip("/").removesuffix("/v1")
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

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read()), (time.monotonic() - t0) * 1000.0

    def props(self):
        with urllib.request.urlopen(self.base + "/props", timeout=self.timeout) as r:
            return json.loads(r.read())

    def tokenize(self, text, add_special=False):
        """special tokens must be parsed, or the chat markers become literal text."""
        out, _ = self._post("/tokenize", {"content": text, "add_special": add_special,
                                          "parse_special": True})
        return out["tokens"]

    def score(self, prompt, label_ids):
        """probabilities for the given label token ids at the next position.

        `prompt` is a token id list or the prompt text. text costs no
        `/tokenize` round trip and is byte-equivalent whenever the spec's
        boundary assertion holds, which is what makes it safe to prefer.

        widens the readout until every label is present. a label missing from a
        truncated candidate list is an artifact, not a zero, and renormalising
        over the survivors looks like a confident answer without being one.
        """
        ladder = spec.constants()["n_probs_ladder"]
        wall, retries = 0.0, 0
        by_id, resp, entries, n_probs = {}, None, [], ladder[0]

        for n_probs in ladder:
            resp, ms = self._post("/completion", {
                "prompt": prompt, "n_predict": 1, "n_probs": n_probs,
                "cache_prompt": self.cache_prompt, "post_sampling_probs": False,
                "temperature": -1.0})
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
