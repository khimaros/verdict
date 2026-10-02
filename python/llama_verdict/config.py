"""config from the environment and a .env file.

the .env is parsed here, in python, rather than by make, so the endpoint,
the tests and the scripts read the same config however they were launched.
the file is plain KEY=value lines: comments start with #, blank lines and
lines without an = are ignored, and nothing is expanded. an empty value
counts as unset, so a tool that exports its own unset variables does not
hide the fallback.

precedence is flags, then the real environment, then the .env. the file is
found by walking up from the working directory, which is both how a server
run from the repo root finds it and how a test runs beside a scratch .env of
its own.
"""

import os

# the backend's address, model and key: verdict's own names, then the ones an
# openai-compatible client conventionally reads (OPENAI_API_BASE is the older
# spelling of OPENAI_BASE_URL)
STANDARD_URLS = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
BACKEND_URLS = ("LLAMA_VERDICT_URL", *STANDARD_URLS)
BACKEND_MODELS = ("LLAMA_VERDICT_MODEL", "OPENAI_MODEL")
BACKEND_KEY, STANDARD_KEY = "LLAMA_VERDICT_API_KEY", "OPENAI_API_KEY"

def dotenv_path(start=None):
    """the nearest .env, searched from the working directory upward."""
    d = os.path.abspath(start or os.getcwd())
    while True:
        candidate = os.path.join(d, ".env")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def parse(text):
    """the variables in .env text. an empty value is no value."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            values[key] = value
    return values


def file_values(path=None):
    """the variables in one .env, or none at all."""
    path = path or dotenv_path()
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return parse(f.read())


def get(name, default=None):
    """one config value: the environment first, then the .env."""
    value = os.environ.get(name)
    if value:
        return value
    return file_values().get(name, default)


def first(names):
    """the first of several settings that is set, in the order given."""
    return next((value for value in map(get, names) if value), None)


def backend_url():
    """the llama-server or llama-swap verdict sits in front of.

    verdict's own name wins wherever it is set, then the names every client of
    an openai-compatible server reads. the order is by name and not by where a
    value came from: OPENAI_BASE_URL is often exported for some other tool,
    and it must not take over a checkout whose .env names its backend.
    """
    return first(BACKEND_URLS)


def backend_model():
    return first(BACKEND_MODELS)


def backend_key(url):
    """the bearer key to present to the backend at `url`, or None.

    verdict's own key goes wherever verdict is pointed. OPENAI_API_KEY is
    usually a real key for somebody else's service, so it goes only to the
    backend the standard url variables name, and never to one that
    LLAMA_VERDICT_URL or a flag named.
    """
    own = get(BACKEND_KEY)
    if own or not url:
        return own

    def origin(address):
        return address.rstrip("/").removesuffix("/v1")

    named = {origin(value) for value in map(get, STANDARD_URLS) if value}
    return get(STANDARD_KEY) if origin(url) in named else None


def backend_url_argument(ap, flag="--base-url"):
    """the backend a script talks to: the flag, else the configured one.

    the address of anyone's eval server is theirs, so no script carries one
    as a default; a .env names it once for every script and the server alike.
    """
    url = backend_url()
    ap.add_argument(flag, default=url, required=url is None,
                    help=f"llama-server or llama-swap base url [{', '.join(BACKEND_URLS)}]")
