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

# Full CI-equivalent gate, including compile, regular pytest, coverage threshold,
# ResourceWarning-visible pytest, config validation, fake verification, and
# desktop static checks when Node is available locally.
ci: lint format-check typecheck
	$(PY) -m compileall -q april_common apps services agents skills tests
	$(PY) -m pytest -q -ra
	$(PY) -m pytest --cov=april_common --cov=apps --cov=services --cov=agents --cov=skills --cov-report=term-missing --cov-fail-under=85
	$(PY) -W default::ResourceWarning -m pytest -q -ra
	$(PY) -m apps.runner.main april config validate
	APRIL_RUNTIME_BACKEND=fake $(PY) -m apps.runner.main april verify --fake
	@if command -v node >/dev/null 2>&1; then \
		node tests/js/desktop_token_bridge.test.cjs; \
		node tests/js/desktop_dashboard.test.cjs; \
		node --check apps/desktop/web/app.js; \
		node --check apps/desktop/web/token_bridge.js; \
		node --check apps/desktop/web/dashboard_helpers.js; \
	else \
		echo "Skipping desktop JS checks: node not found."; \
	fi

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
