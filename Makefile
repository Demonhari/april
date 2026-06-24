PYTHON ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: install install-dev test coverage lint format format-check typecheck check ci run-runtime run-api cli install-global install-global-path install-global-force install-global-force-path verify-global uninstall-global

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e .

install-dev:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e '.[dev]' -c constraints-dev.txt

test:
	$(PY) -m pytest

coverage:
	$(PY) -m pytest --cov=april_common --cov=services --cov=agents --cov=skills --cov=apps --cov-report=term-missing --cov-fail-under=85

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

format-check:
	$(PY) -m ruff format --check .

typecheck:
	$(PY) -m mypy april_common apps services agents skills

# Mirror the important CI quality gates: lint, format check, type check, tests.
check: lint format-check typecheck test

# Full CI-equivalent gate, including coverage threshold, compile, config
# validation and the fake-backend verification run.
ci: lint format-check typecheck coverage
	$(PY) -m compileall -q april_common apps services agents skills tests
	$(PY) -m apps.runner.main april config validate
	$(PY) -m apps.runner.main april verify --fake

run-runtime:
	$(PY) -m services.april_runtime.server

run-api:
	$(PY) -m services.api.server

cli:
	$(PY) -m apps.cli.main

install-global:
	bash scripts/install_run_april.sh

install-global-path:
	bash scripts/install_run_april.sh --add-to-path

install-global-force:
	bash scripts/install_run_april.sh --force

install-global-force-path:
	bash scripts/install_run_april.sh --force --add-to-path

verify-global:
	"$(HOME)/.local/bin/run" april status

uninstall-global:
	bash scripts/uninstall_run_april.sh
