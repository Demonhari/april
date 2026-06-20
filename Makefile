PYTHON ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: install install-dev test coverage lint format typecheck check run-runtime run-api cli install-global install-global-path install-global-force install-global-force-path verify-global uninstall-global

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e .

install-dev:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e '.[dev]'

test:
	$(PY) -m pytest

coverage:
	$(PY) -m pytest --cov=april_common --cov=services --cov=agents --cov=skills --cov=apps --cov-report=term-missing

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

typecheck:
	$(PY) -m mypy april_common apps services agents skills

check: lint typecheck test

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
