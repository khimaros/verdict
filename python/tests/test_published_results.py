"""nothing in the repo names the machines it was measured on.

eval/results is committed and read by the model registry, and the scripts and
docs are public. a log line that echoes an absolute path publishes a home
directory; a default that names the eval server publishes a private address.
the evidence that matters -- answers, settings, weights -- never needs either.
"""

import os
import re
import urllib.parse

from llama_verdict import config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv"}
# .env holds the local config and is ignored by git, so it may name anything
SKIP_FILES = {".env"}
LOCAL_PATH = re.compile(r"(?<![\w/])/(home|Users|root)/[\w.-]+")
PRIVATE_IP = re.compile(r"(?<![\d.])(10\.\d{1,3}|192\.168|172\.(1[6-9]|2\d|3[01]))"
                        r"\.\d{1,3}\.\d{1,3}(?![\d.])")


def text_files():
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name in SKIP_FILES:
                continue
            path = os.path.join(folder, name)
            with open(path, "rb") as f:
                data = f.read()
            if b"\0" not in data:
                yield os.path.relpath(path, ROOT), data.decode(errors="replace")


def offending(pattern):
    return [path for path, text in text_files() if pattern.search(text)]


def test_nothing_names_a_local_path():
    assert not offending(LOCAL_PATH)


def test_nothing_names_a_private_address():
    assert not offending(PRIVATE_IP)


def test_nothing_names_the_configured_backend_host():
    url = config.get("LLAMA_VERDICT_URL")
    host = urllib.parse.urlsplit(url).hostname if url else None
    if not host or host in ("localhost", "127.0.0.1"):
        return
    assert not offending(re.compile(re.escape(host)))
