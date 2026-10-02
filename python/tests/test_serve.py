"""the serve target and the config the endpoint reads.

the endpoint is meant to take its model, key and port from a `.env` rather
than from a shell invocation long enough to mistype, and that .env is parsed
by the python code, not by make: the server, the tests and the scripts then
read the same config however they were launched. precedence is flags, then
the environment, then the nearest .env, searched upward from the working
directory, which is what lets these cases run in a scratch directory holding
no .env at all. the make recipes are only ever echoed or stopped, so nothing
here needs weights.
"""

import importlib
import os
import shutil
import subprocess
import sys

import pytest

from llama_verdict import config, server, spec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAKEFILE = os.path.join(ROOT, "Makefile")

# the knobs .env is expected to carry, and the name each one answers to
CONFIG = ("LLAMA_VERDICT_URL", "LLAMA_VERDICT_MODEL", "LLAMA_VERDICT_SPEC",
          "LLAMA_VERDICT_API_KEY", "VERDICT_API_KEY", "VERDICT_HOST", "VERDICT_PORT",
          "VERDICT_FORMATTER", "VERDICT_ASSISTANT_OPEN",
          "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY", "OPENAI_MODEL")


def make(tmp_path, target, args=(), env=None):
    """run a make target against a copy of the real Makefile."""
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    child = {k: v for k, v in os.environ.items() if k not in CONFIG}
    child.update(env or {})
    return subprocess.run(
        ["make", "-C", str(tmp_path), *args, target],
        capture_output=True, text=True, env=child)


def isolated(tmp_path, monkeypatch):
    """strip the config from the environment and work where no .env exists."""
    for name in CONFIG:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def dotenv(tmp_path, text):
    (tmp_path / ".env").write_text(text)


def test_a_dotenv_file_configures_the_endpoint(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    dotenv(tmp_path, "LLAMA_VERDICT_URL=http://box:8080/v1\n"
                     "LLAMA_VERDICT_MODEL=qwen3.5-9b:Q8_0\n"
                     "VERDICT_PORT=9001\n")
    args = server.parse_args([])
    assert args.base_url == "http://box:8080/v1"
    assert args.model == "qwen3.5-9b:Q8_0"
    assert args.port == 9001


def test_a_dotenv_higher_up_the_tree_is_found(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    dotenv(tmp_path, "LLAMA_VERDICT_URL=http://box:8080/v1\n")
    nested = tmp_path / "python" / "tests"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert config.get("LLAMA_VERDICT_URL") == "http://box:8080/v1"


def test_the_environment_beats_the_dotenv(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    dotenv(tmp_path, "LLAMA_VERDICT_MODEL=from-the-dotenv\n"
                     "LLAMA_VERDICT_URL=http://box:8080/v1\n")
    monkeypatch.setenv("LLAMA_VERDICT_MODEL", "from-the-environment")
    assert server.parse_args([]).model == "from-the-environment"


def test_flags_beat_the_dotenv(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    dotenv(tmp_path, "LLAMA_VERDICT_MODEL=from-the-dotenv\n"
                     "LLAMA_VERDICT_URL=http://box:8080/v1\n")
    args = server.parse_args(["--model", "from-the-flag"])
    assert args.model == "from-the-flag"


def dotenv_comments_case(tmp_path):
    dotenv(tmp_path, "# a comment\n"
                     "\n"
                     "  # indented comment\n"
                     "NOT_A_LINE\n"
                     "LLAMA_VERDICT_MODEL = spaced : name \n"
                     "LLAMA_VERDICT_SPEC=\n")


def test_comments_and_blanks_are_not_variables(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    dotenv_comments_case(tmp_path)
    assert config.get("NOT_A_LINE") is None
    # spaces around the = are tolerated and trimmed; the file is data, not a
    # shell script, and nothing here ever expands a value
    assert config.get("LLAMA_VERDICT_MODEL") == "spaced : name"


def test_an_empty_value_counts_as_unset(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    dotenv_comments_case(tmp_path)
    # a tool that exports its own unset variables must not hide the fallback,
    # and `KEY=` in a .env means the same thing
    monkeypatch.setenv("LLAMA_VERDICT_SPEC", "")
    assert config.get("LLAMA_VERDICT_SPEC") is None


def test_the_server_reads_its_config_from_the_environment(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("LLAMA_VERDICT_URL", "http://box:8080/v1")
    monkeypatch.setenv("LLAMA_VERDICT_MODEL", "qwen3.5-9b:Q8_0")
    monkeypatch.setenv("VERDICT_PORT", "9001")
    monkeypatch.setenv("VERDICT_FORMATTER", "spec/formatters/golden.json")
    args = server.parse_args([])
    assert args.base_url == "http://box:8080/v1"
    assert args.model == "qwen3.5-9b:Q8_0"
    assert args.port == 9001
    assert args.formatter == "spec/formatters/golden.json"


def test_the_assistant_opening_is_read_from_the_variable_its_help_names(
        tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("LLAMA_VERDICT_URL", "http://box:8080/v1")
    monkeypatch.setenv("LLAMA_VERDICT_MODEL", "gpt-oss-20b:Q8_0")
    monkeypatch.setenv("VERDICT_ASSISTANT_OPEN", "<|start|>assistant")
    assert server.parse_args([]).assistant_open == "<|start|>assistant"
    assert server.parse_args(["--assistant-open", "<|a|>"]).assistant_open == "<|a|>"


def test_flags_beat_the_environment(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("LLAMA_VERDICT_MODEL", "from-the-environment")
    monkeypatch.setenv("LLAMA_VERDICT_URL", "http://box:8080/v1")
    args = server.parse_args(["--model", "from-the-command-line"])
    assert args.model == "from-the-command-line"


def test_the_standard_openai_names_configure_the_backend(tmp_path, monkeypatch):
    """the backend is an openai-compatible server, so the names every other
    client of one already reads are enough to point verdict at it."""
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://box:8080/v1")
    monkeypatch.setenv("OPENAI_MODEL", "qwen3.5-9b:Q8_0")
    args = server.parse_args([])
    assert (args.base_url, args.model) == ("http://box:8080/v1", "qwen3.5-9b:Q8_0")


def test_the_older_spelling_of_the_base_url_is_read_too(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://box:8080/v1")
    assert config.backend_url() == "http://box:8080/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://newer:8080/v1")
    assert config.backend_url() == "http://newer:8080/v1"


def test_verdicts_own_names_win_over_the_standard_ones(tmp_path, monkeypatch):
    """wherever they are set. OPENAI_BASE_URL is often exported for some other
    tool, and it must not take over a checkout whose .env names its backend."""
    isolated(tmp_path, monkeypatch)
    dotenv(tmp_path, "LLAMA_VERDICT_URL=http://box:8080/v1\n"
                     "LLAMA_VERDICT_MODEL=from-the-dotenv\n")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.example/v1")
    monkeypatch.setenv("OPENAI_MODEL", "something-else")
    args = server.parse_args([])
    assert (args.base_url, args.model) == ("http://box:8080/v1", "from-the-dotenv")


def test_the_standard_key_goes_only_to_the_backend_the_standard_url_names(
        tmp_path, monkeypatch):
    """OPENAI_API_KEY is usually a real key for somebody else's service. it is
    presented to the backend only when OPENAI_BASE_URL is what named that
    backend, never to one named by verdict's own setting or by a flag."""
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-standard")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://box:8080/v1")
    assert config.backend_key("http://box:8080/v1") == "sk-standard"
    assert config.backend_key("http://box:8080") == "sk-standard"
    assert config.backend_key("http://another-box:8080/v1") is None
    monkeypatch.setenv("LLAMA_VERDICT_URL", "http://another-box:8080/v1")
    assert config.backend_key(config.backend_url()) is None


def test_verdicts_own_backend_key_goes_wherever_verdict_is_pointed(tmp_path, monkeypatch):
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("LLAMA_VERDICT_API_KEY", "own-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-standard")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://box:8080/v1")
    assert config.backend_key("http://box:8080/v1") == "own-key"
    assert config.backend_key("http://anywhere:1/v1") == "own-key"


def test_a_missing_backend_names_the_variables(tmp_path, monkeypatch, capsys):
    isolated(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        server.parse_args([])
    err = capsys.readouterr().err
    assert "LLAMA_VERDICT_URL" in err
    # the answer is a .env, so say where that goes and what belongs in it
    assert ".env.example" in err


def test_a_backend_is_enough_and_a_bound_model_is_optional(tmp_path, monkeypatch):
    """an endpoint that routes answers for whichever model a request names, so
    the bound one is only what the aliases mean."""
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("LLAMA_VERDICT_URL", "http://box:8080/v1")
    assert server.parse_args([]).model is None


def test_an_endpoint_that_neither_routes_nor_binds_a_model_is_refused(
        tmp_path, monkeypatch, capsys):
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("LLAMA_VERDICT_URL", "http://box:8080/v1")
    with pytest.raises(SystemExit):
        server.parse_args(["--no-routing"])
    assert "LLAMA_VERDICT_MODEL" in capsys.readouterr().err


def test_a_flag_about_the_bound_model_needs_one(tmp_path, monkeypatch, capsys):
    isolated(tmp_path, monkeypatch)
    monkeypatch.setenv("LLAMA_VERDICT_URL", "http://box:8080/v1")
    with pytest.raises(SystemExit):
        server.parse_args(["--layout", "decider-plain"])
    assert "--model" in capsys.readouterr().err


def test_a_dotenv_file_configures_a_server_run_from_the_shell(tmp_path):
    # the whole point of parsing .env in python: the documented one-liner,
    # with no make and no exports, reads the config off disk like make used to
    isolated_dir = tmp_path / "workdir"
    isolated_dir.mkdir()
    (isolated_dir / ".env").write_text(
        "LLAMA_VERDICT_URL=http://127.0.0.1:1/v1\n"
        "LLAMA_VERDICT_MODEL=nope:Q8_0\n")
    child = {k: v for k, v in os.environ.items() if k not in CONFIG}
    child["PYTHONPATH"] = os.path.join(ROOT, "python")
    out = subprocess.run([sys.executable, "-m", "llama_verdict.server"],
                         cwd=isolated_dir, env=child, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode != 0
    # it got far enough to try the backend the .env named and fail on it
    assert "cannot reach the backend" in out.stderr
    assert "127.0.0.1:1" in out.stderr


def test_serve_starts_the_reference_endpoint(tmp_path):
    out = make(tmp_path, "serve", args=("-n",))
    assert out.returncode == 0, out.stderr
    assert "llama_verdict.server" in out.stdout


def test_serve_is_documented(tmp_path):
    assert "serve" in make(tmp_path, "help").stdout


def test_an_unreachable_backend_is_reported_plainly(tmp_path, monkeypatch, capsys):
    isolated(tmp_path, monkeypatch)
    # a refused connection is the first thing `make serve` hits when the
    # backend is down, and a traceback is not an answer to that
    with pytest.raises(SystemExit) as exc:
        server.main(["--base-url", "http://127.0.0.1:1/v1", "--model", "nope:Q8_0"])
    assert "cannot reach the backend" in str(exc.value)
    assert "127.0.0.1:1" in str(exc.value)
    assert "Traceback" not in capsys.readouterr().err


def test_an_empty_spec_override_falls_back(monkeypatch):
    monkeypatch.setenv("LLAMA_VERDICT_SPEC", "")
    try:
        assert os.path.isdir(importlib.reload(spec).SPEC_DIR)
    finally:
        importlib.reload(spec)


def test_env_example_documents_the_variables():
    with open(os.path.join(ROOT, ".env.example")) as f:
        example = f.read()
    assert all(name in example for name in CONFIG), example
