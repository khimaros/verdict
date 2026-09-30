"""the endpoint is runnable with plain `uv run`, no PYTHONPATH and no venv
dance.

the package lives in python/ and is the uv project; `uv run` from there
installs it editable and runs the module against a uv-managed interpreter.
these cases prove the project metadata actually supports that -- it once did
not, because the pyproject pointed its readme outside the project and no
build backend may package files from above its own directory.

skips when uv is not installed. the first run on a fresh checkout syncs the
project environment and needs a network for the build backend; after that
uv runs offline.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PY_DIR = ROOT / "python"

UV = shutil.which("uv")

pytestmark = pytest.mark.skipif(not UV, reason="uv not installed")


def uv_run(*args, cwd=None):
    # the config stripped: the server must fail for the documented reason,
    # not because this shell happens to export a backend
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LLAMA_VERDICT_", "VERDICT_"))}
    return subprocess.run(
        [UV, "run", *args], cwd=cwd or PY_DIR, env=env,
        capture_output=True, text=True, timeout=900)


def test_uv_run_starts_the_server_module():
    out = uv_run("python", "-m", "llama_verdict.server", "--help")
    assert out.returncode == 0, out.stderr
    assert "--base-url" in out.stdout
    # the config story has to survive the move to uv: no make, no exports
    assert "LLAMA_VERDICT_URL" in out.stdout


def test_uv_run_can_derive_a_formatter():
    # deriving from the model's own template is the normal startup path, so
    # jinja2 rides in the default environment rather than behind an extra
    # the runner has to know about
    out = uv_run("python", "-c", "import jinja2")
    assert out.returncode == 0, out.stderr


def test_uv_run_imports_the_package_from_anywhere(tmp_path):
    # a real console invocation: cwd outside the project, config from the
    # environment rather than a .env, and the endpoint's argparse contract
    # holds exactly as the docs promise
    out = uv_run("--project", str(PY_DIR), "python", "-m",
                 "llama_verdict.server", cwd=tmp_path)
    assert out.returncode != 0
    assert "LLAMA_VERDICT_URL" in out.stderr
    assert ".env.example" in out.stderr
