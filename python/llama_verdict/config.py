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


def backend_url_argument(ap, flag="--base-url"):
    """the backend a script talks to: the flag, else LLAMA_VERDICT_URL.

    the address of anyone's eval server is theirs, so no script carries one
    as a default; a .env names it once for every script and the server alike.
    """
    url = get("LLAMA_VERDICT_URL")
    ap.add_argument(flag, default=url, required=url is None,
                    help="llama-server or llama-swap base url [LLAMA_VERDICT_URL]")
