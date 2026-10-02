"""clef's joint schema head: every question of a request decided in one pass.

unlike a linear head, this one reads the final hidden state of EVERY position.
each option is a span of the prompt, scored from the mean hidden state over
that span, the backbone's output embedding rows for its tokens, and evidence
routed to it from the whole sequence; the questions then attend to each other
before any is scored. spec/SPEC.md 5.5.

a port of `JointSchemaHead.forward` in Cloudflare/clef-flash @ 17f0b0a
`joint_schema_model.py`, which is torch and the only published implementation.
it is held to that class's own logits by `tests/test_joint_head.py`.

numpy is imported here and nowhere else in the package. six attention layers
over thousands of positions are not stdlib work, and only a model read through
this head pays for the dependency, as only derivation pays for jinja2.
"""

import functools
import hashlib
import json
import math
import os
import urllib.request

import numpy as np

from . import head as heads

# torch's defaults, which the author's modules were built with
LAYER_NORM_EPS = 1e-5
NORMALIZE_EPS = 1e-12
COSINE_EPS = 1e-8
# the author clamps both learned scales before exponentiating them
MAX_LOG_SCALE = math.log(100.0)
SAFETENSORS_DTYPES = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}

erf = np.vectorize(math.erf, otypes=[np.float32])


def widen(raw, dtype):
    """raw safetensors values as float32."""
    if dtype == "BF16":
        # the top half of a float32, so widening is exact
        return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
    return np.frombuffer(raw, dtype=SAFETENSORS_DTYPES[dtype]).astype(np.float32)


def read_safetensors(path):
    """every tensor of a safetensors file as float32: an 8 byte header length,
    a json header of dtype, shape and byte range per tensor, the raw values."""
    with open(path, "rb") as f:
        data = f.read()
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + size])
    return {name: widen(data[8 + size + meta["data_offsets"][0]:
                             8 + size + meta["data_offsets"][1]],
                        meta["dtype"]).reshape(meta["shape"])
            for name, meta in header.items() if name != "__metadata__"}


class Rows:
    """rows of one tensor in a safetensors file on the hub, by index.

    a joint head scores each option partly from the backbone's output
    embedding rows for the option's tokens. no llama-server endpoint serves
    that matrix, and the gguf's copy is quantised where the head was trained
    against bf16, so the rows are read from the author's own weights: a byte
    range per run of rows, never the multi-gigabyte file, and each row kept on
    disk once fetched. the revision pins the bytes.
    """

    def __init__(self, declared, cache_dir=None, path=None):
        """`path` reads a safetensors file on disk in place of the hub's."""
        self.url, self.tensor = heads.HUB_URL.format(**declared), declared["tensor"]
        self.path = path
        source = (hashlib.sha256(os.path.abspath(path).encode()).hexdigest() if path
                  else f"{declared['revision']}-{os.path.basename(declared['file'])}")
        self.dir = os.path.join(cache_dir or heads.CACHE_DIR, "rows", f"{source}-{self.tensor}")
        self._where = None

    def _range(self, start, end):
        if self.path:
            with open(self.path, "rb") as f:
                f.seek(start)
                return f.read(end - start)
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=heads.FETCH_TIMEOUT) as r:
            return r.read()

    def _located(self):
        """the tensor's dtype, row width in bytes and offset in the file."""
        path = os.path.join(self.dir, "header.json")
        if self._where is None and os.path.exists(path):
            with open(path) as f:
                self._where = json.load(f)
        if self._where is None:
            size = int.from_bytes(self._range(0, 8), "little")
            meta = json.loads(self._range(8, 8 + size))[self.tensor]
            self._where = {
                "dtype": meta["dtype"], "start": 8 + size + meta["data_offsets"][0],
                "row_bytes": (meta["data_offsets"][1] - meta["data_offsets"][0])
                // meta["shape"][0]}
            os.makedirs(self.dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._where, f)
        return self._where

    def _fetch(self, first, last):
        """rows first..last inclusive, in one request, each written to disk."""
        w = self._located()
        raw = self._range(w["start"] + first * w["row_bytes"],
                          w["start"] + (last + 1) * w["row_bytes"])
        for n, index in enumerate(range(first, last + 1)):
            widen(raw[n * w["row_bytes"]:(n + 1) * w["row_bytes"]], w["dtype"]).tofile(
                os.path.join(self.dir, f"{index}.f32"))

    def __call__(self, indices):
        missing = sorted(i for i in set(indices)
                         if not os.path.exists(os.path.join(self.dir, f"{i}.f32")))
        while missing:
            run = 1
            while run < len(missing) and missing[run] == missing[0] + run:
                run += 1
            self._fetch(missing[0], missing[run - 1])
            missing = missing[run:]
        return np.stack([np.fromfile(os.path.join(self.dir, f"{i}.f32"), dtype=np.float32)
                         for i in indices])


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), NORMALIZE_EPS)


@functools.cache
def tensors_of(path):
    return read_safetensors(path)


def load(path, declared, rows_path=None):
    """the head a layout declares: its weights, its attention head count and
    where the embedding rows it reads come from."""
    return JointSchema(tensors_of(path), declared["heads"],
                       Rows(declared["rows"], path=rows_path))


class JointSchema:
    joint = True

    def __init__(self, tensors, heads, rows=None):
        self.t, self.heads, self.rows = tensors, heads, rows
        self.width = tensors["memory_projection.weight"].shape[0]

    def _layers(self, prefix):
        return sorted({name.split(".")[1] for name in self.t if name.startswith(prefix + ".")},
                      key=int)

    def _linear(self, name, x):
        bias = self.t.get(f"{name}.bias")
        return x @ self.t[f"{name}.weight"].T + (0.0 if bias is None else bias)

    def _norm(self, name, x):
        mean = x.mean(axis=-1, keepdims=True)
        spread = np.sqrt(x.var(axis=-1, keepdims=True) + LAYER_NORM_EPS)
        return (x - mean) / spread * self.t[f"{name}.weight"] + self.t[f"{name}.bias"]

    def _attend(self, name, queries, memory):
        """multi-head attention of `queries` over `memory`, unmasked."""
        e, d = self.width, self.width // self.heads
        weight, bias = self.t[f"{name}.in_proj_weight"], self.t[f"{name}.in_proj_bias"]

        def split(x, part):
            projected = x @ weight[part * e:(part + 1) * e].T + bias[part * e:(part + 1) * e]
            return projected.reshape(len(x), self.heads, d).transpose(1, 0, 2)

        q, k, v = split(queries, 0), split(memory, 1), split(memory, 2)
        mixed = softmax(q @ k.transpose(0, 2, 1) / math.sqrt(d)) @ v
        return self._linear(f"{name}.out_proj", mixed.transpose(1, 0, 2).reshape(len(queries), e))

    def _route(self, layer, options, memory):
        """one evidence routing layer: each option gathers from the sequence."""
        p = f"evidence_layers.{layer}"
        options = options + self._attend(f"{p}.attention", self._norm(f"{p}.query_norm", options),
                                         self._norm(f"{p}.memory_norm", memory))
        normed = self._norm(f"{p}.feedforward_norm", options)
        return options + self._linear(f"{p}.feedforward.3",
                                      gelu(self._linear(f"{p}.feedforward.0", normed)))

    def _decode(self, layer, fields, memory):
        """one norm-first decoder layer: the questions attend to each other,
        then to the sequence."""
        p = f"layers.{layer}"
        own = self._norm(f"{p}.norm1", fields)
        fields = fields + self._attend(f"{p}.self_attn", own, own)
        fields = fields + self._attend(f"{p}.multihead_attn", self._norm(f"{p}.norm2", fields),
                                       memory)
        return fields + self._linear(f"{p}.linear2", gelu(self._linear(
            f"{p}.linear1", self._norm(f"{p}.norm3", fields))))

    def _score(self, field, options, lexical, anchor):
        """one question's logits: a lexical prior plus a gated joint score."""
        scale = {name: math.exp(min(float(self.t[name]), MAX_LOG_SCALE))
                 for name in ("prior_logit_scale", "joint_logit_scale")}
        options = self._norm("option_norm", options)
        repeated = np.broadcast_to(field, options.shape)
        cosine = (repeated * options).sum(-1) / (
            np.maximum(np.linalg.norm(repeated, axis=-1), COSINE_EPS)
            * np.maximum(np.linalg.norm(options, axis=-1), COSINE_EPS))
        features = np.concatenate(
            [repeated, options, repeated * options, np.abs(repeated - options)], axis=-1)
        residual = self._linear("residual_scorer.3", gelu(self._linear(
            "residual_scorer.0", features)))[:, 0]
        gate = 1.0 / (1.0 + math.exp(-float(self.t["residual_gate"])))
        return (scale["prior_logit_scale"] * (unit(lexical) @ anchor)
                + gate * (scale["joint_logit_scale"] * cosine + residual))

    def logits(self, hidden, token_ids, questions, rows=None):
        """logits per question, one per option.

        `hidden` is the final hidden state of every prompt position, `rows`
        returns the backbone's output embedding rows for a list of token ids.
        """
        rows = rows or self.rows
        hidden = self._norm("hidden_norm", np.asarray(hidden, dtype=np.float32))
        memory = self._linear("memory_projection", hidden)
        asked = np.stack([hidden[slice(*q.question)].mean(axis=0) for q in questions])
        lexical = [np.stack([rows(list(token_ids[s:e])).mean(axis=0) for s, e in q.options])
                   for q in questions]
        queries = [
            self._linear("option_context_projection",
                         np.stack([hidden[s:e].mean(axis=0) for s, e in q.options]))
            + self._linear("option_lexical_projection", lex)
            + self._linear("option_question_projection", asked[i])
            for i, (q, lex) in enumerate(zip(questions, lexical, strict=True))]
        routed = np.concatenate(queries)
        for layer in self._layers("evidence_layers"):
            routed = self._route(layer, routed, memory)
        options = np.split(routed, np.cumsum([len(q.options) for q in questions])[:-1])

        base = self._linear("question_projection", asked)
        summaries = np.stack([softmax(o @ f / math.sqrt(self.width)) @ o
                              for f, o in zip(base, options, strict=True)])
        fields = (base + self._norm("option_summary_norm", summaries)
                  + self._linear("global_projection", hidden[-1])
                  + self.t["type_embedding.weight"][[q.kind for q in questions]])
        for layer in self._layers("layers"):
            fields = self._decode(layer, fields, memory)
        fields = self._norm("field_norm", fields)
        return [self._score(fields[i], options[i], lexical[i], unit(asked[i] + hidden[-1]))
                for i in range(len(questions))]

    @staticmethod
    def probabilities(logits):
        return [softmax(row).tolist() for row in logits]
