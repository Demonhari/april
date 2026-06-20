#!/usr/bin/env bash
set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
elif [[ "${1:-}" != "" ]]; then
  echo "Usage: scripts/install_run_april.sh [--force]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/pip install -e ".[dev]"

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

ARGS=(--install --repo-root "$REPO_ROOT" --bin-dir "$BIN_DIR")
if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

.venv/bin/python -m apps.runner.install "${ARGS[@]}"
