PYTHON ?= python3
PY_DIR := python
SPEC_DIR := spec

# a live llama-server for the targets that need real weights. llama-swap serves
# the native endpoints under /upstream/<model>/, which the client handles.
LLAMA_VERDICT_URL ?=
LLAMA_VERDICT_MODEL ?=

export PYTHONPATH := $(CURDIR)/$(PY_DIR)

.PHONY: build test test-e2e precommit lint fixtures formatters clean help

build: ## validate the spec and the generated artifacts
	@$(PYTHON) -c "import json,glob; [json.load(open(p)) for p in \
	  glob.glob('$(SPEC_DIR)/**/*.json', recursive=True)]" && echo "spec json ok"
	@$(PYTHON) scripts/build_fixtures.py --check

test: ## conformance tests against spec/fixtures
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests/

test-e2e: ## end to end against a live llama-server; needs LLAMA_VERDICT_URL
ifeq ($(strip $(LLAMA_VERDICT_URL)),)
	@echo "skipping: set LLAMA_VERDICT_URL (and LLAMA_VERDICT_MODEL behind llama-swap)"
else
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests_e2e/ -v
endif

lint: ## static analysis
	@if command -v ruff >/dev/null 2>&1; then RUFF="ruff"; \
	 elif command -v uvx >/dev/null 2>&1; then RUFF="uvx ruff"; \
	 else echo "!!! ruff not found; install it or make uvx available"; exit 1; fi; \
	 $$RUFF check $(PY_DIR) scripts

precommit: lint build test ## everything that must pass before a commit

fixtures: ## regenerate the conformance fixtures, then review the diff
	@$(PYTHON) scripts/build_fixtures.py

formatters: ## derive formatters from a live server; needs LLAMA_VERDICT_URL
ifeq ($(strip $(LLAMA_VERDICT_URL)),)
	@echo "set LLAMA_VERDICT_URL and LLAMA_VERDICT_MODEL"
else
	@$(PYTHON) scripts/derive_formatter.py --base-url $(LLAMA_VERDICT_URL) \
	  --models $(LLAMA_VERDICT_MODEL)
endif

clean:
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; true
	@rm -rf $(PY_DIR)/.pytest_cache

help:
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'
