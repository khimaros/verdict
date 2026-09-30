PYTHON ?= python3
PY_DIR := python
SPEC_DIR := spec

export PYTHONPATH := $(CURDIR)/$(PY_DIR)

# dev config lives in a .env at the repo root and the python code parses it,
# not make: serve, test-e2e and formatters need no arguments and no exports.
# the python side reads the environment first and the nearest .env second;
# see python/llama_verdict/config.py and .env.example.

.PHONY: build test test-e2e serve precommit lint fixtures formatters export clean help

build: ## validate the spec and the generated artifacts
	@$(PYTHON) -c "import json,glob; [json.load(open(p)) for p in \
	  glob.glob('$(SPEC_DIR)/**/*.json', recursive=True)]" && echo "spec json ok"
	@$(PYTHON) scripts/build_fixtures.py --check

test: ## conformance tests against spec/fixtures
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests/

# both suites live in tests_e2e and skip themselves: the live one needs the
# backend from the environment or .env, the mock one needs the fake-openai
# binary built (make -C ../fake-openai). running the directory always means the
# weightless suite runs on a machine with neither the server nor a hint of one.
test-e2e: ## end to end; the live suite needs a backend, the mock suite needs ../fake-openai
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests_e2e/ -v

serve: ## launch the jev endpoint; needs the backend from .env or the environment
	@$(PYTHON) -m llama_verdict.server

lint: ## static analysis
	@if command -v ruff >/dev/null 2>&1; then RUFF="ruff"; \
	 elif command -v uvx >/dev/null 2>&1; then RUFF="uvx ruff"; \
	 else echo "!!! ruff not found; install it or make uvx available"; exit 1; fi; \
	 $$RUFF check $(PY_DIR) scripts

precommit: lint build test ## everything that must pass before a commit

fixtures: ## regenerate the conformance fixtures, then review the diff
	@$(PYTHON) scripts/build_fixtures.py

formatters: ## derive formatters from a live server; needs the backend from .env
	@$(PYTHON) scripts/derive_formatter.py

export: ## publish every measurement in eval/results as eval/results/export.json
	@$(PYTHON) scripts/export_results.py

clean:
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; true
	@rm -rf $(PY_DIR)/.pytest_cache

help:
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'
