"""a trained decision head, applied to the hidden state at the position read.

some decision models do not answer through their vocabulary. they were trained
with a head of their own over the final hidden state, and a layout that names
one (spec/SPEC.md 5.4) is read by fetching that hidden state from a
llama-server in embedding mode and applying the head here.

stdlib only, which is why the weights are parsed by hand: an `.npz` is a zip of
`.npy` arrays and a float32 `.npy` is a short header followed by the raw
values. a head too large for that belongs to a head kind of its own.
"""

import array
import ast
import functools
import hashlib
import math
import operator
import os
import struct
import sys
import urllib.request
import zipfile

from . import spec
from .derive import CACHE_HOME

CACHE_DIR = os.path.join(CACHE_HOME, "verdict", "heads")
HUB_URL = "https://huggingface.co/{repo}/resolve/{revision}/{file}"
FETCH_TIMEOUT = 300
NPY_MAGIC = b"\x93NUMPY"
# where an npy header's length field ends, by format major version
NPY_HEADER = {1: ("<H", 10), 2: ("<I", 12), 3: ("<I", 12)}


def read_npy(data):
    """one float32 array as (shape, values), from numpy's own file format."""
    if data[:6] != NPY_MAGIC or data[6] not in NPY_HEADER:
        raise ValueError("not an npy array")
    fmt, start = NPY_HEADER[data[6]]
    size = struct.unpack_from(fmt, data, 8)[0]
    meta = ast.literal_eval(data[start:start + size].decode("latin-1"))
    if meta["descr"] != "<f4" or meta["fortran_order"]:
        raise ValueError(f"a head is read as row-major float32, not {meta['descr']!r}")
    values = array.array("f")
    values.frombytes(data[start + size:])
    if sys.byteorder == "big":
        values.byteswap()
    return meta["shape"], values


class Linear:
    """`softmax(W[:n] @ ((h - mu) / sd) + b[:n])`, one row per option slot.

    the same shape as reading label logits, which are the vocabulary's rows
    applied to this hidden state, with rows trained for the purpose instead.
    """
    joint = False

    def __init__(self, tensors):
        (rows, self.width), weight = tensors["linear.weight"]
        self.rows = [weight[i * self.width:(i + 1) * self.width] for i in range(rows)]
        self.bias, self.mu, self.sd = (tensors[name][1] for name in ("linear.bias", "mu", "sd"))

    def probs(self, hidden, count):
        """a probability per option, in the order the options were listed."""
        if count > len(self.rows):
            raise ValueError(f"the head has {len(self.rows)} option slots and the "
                             f"question lists {count}. shortlist the options.")
        if len(hidden) != self.width:
            raise ValueError(f"the head reads a hidden state {self.width} wide and the "
                             f"backend returned {len(hidden)}: it is not serving the "
                             f"model this head was trained on.")
        x = [(h - m) / s for h, m, s in zip(hidden, self.mu, self.sd, strict=True)]
        z = [sum(map(operator.mul, row, x)) + b
             for row, b in zip(self.rows[:count], self.bias, strict=False)]
        top = max(z)
        e = [math.exp(v - top) for v in z]
        return [v / sum(e) for v in e]


KINDS = {"linear": Linear}
# a head that decides every question of a request from every position, in
# llama_verdict/joint_head.py
JOINT = "joint-schema"


@functools.cache
def load(path, kind="linear"):
    if kind not in KINDS:
        raise ValueError(f"this verdict does not know the head kind {kind!r}; "
                         f"it applies {sorted([*KINDS, JOINT])}")
    with zipfile.ZipFile(path) as z:
        return KINDS[kind]({name.removesuffix(".npy"): read_npy(z.read(name))
                            for name in z.namelist()})


def fetch(declared, cache_dir=CACHE_DIR):
    """the head a layout pins, downloaded once and held to its checksum.

    fetched rather than vendored (NFR4), and refused on a mismatch: a head
    re-uploaded under the same name would otherwise change every answer while
    the model id stayed the same.
    """
    if not declared.get("sha256"):
        raise ValueError(f"the head {declared.get('file')!r} is pinned to no sha256, so a "
                         f"fetched copy cannot be checked. pass one on disk with --head.")
    path = os.path.join(cache_dir, f"{declared['sha256']}-{os.path.basename(declared['file'])}")
    if os.path.exists(path):
        return path
    url = HUB_URL.format(**declared)
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != declared["sha256"]:
        raise ValueError(f"{url} has sha256 {digest}, and the layout pins {declared['sha256']}")
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def for_layout(name, path=None, rows_path=None):
    """the head a layout is read through, or None for a layout read by logits.
    `path` is a head already on disk, used as it is, and `rows_path` a
    safetensors file on disk holding the embedding rows a joint head reads."""
    declared = name != spec.CHAT and spec.load_layout(name).get("head")
    if not declared:
        return None
    path = path or fetch(declared)
    if declared.get("kind") == JOINT:
        # numpy comes in with this import, and only a model read this way pays
        try:
            from . import joint_head
        except ImportError as e:
            raise ValueError(
                f"layout {name!r} is read through a joint head, which needs numpy, and "
                f"this python has none ({e}). install it (pip install numpy, or the "
                f"llama-verdict[joint-head] extra), or run under the project's "
                f"environment: uv run python -m llama_verdict.server, from python/.") from e
        return joint_head.load(path, declared, rows_path)
    return load(path, declared.get("kind"))
