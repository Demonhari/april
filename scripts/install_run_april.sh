#!/usr/bin/env bash
set -euo pipefail

FORCE=0
ADD_TO_PATH=0

show_help() {
  cat <<'HELP'
Usage: scripts/install_run_april.sh [--force] [--add-to-path]

Installs APRIL-owned wrappers into ~/.local/bin. This script never uses sudo
and only edits shell PATH config when --add-to-path is supplied.
HELP
}

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --add-to-path) ADD_TO_PATH=1 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; show_help >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ "${APRIL_INSTALL_SKIP_PIP:-0}" != "1" && ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [[ "${APRIL_INSTALL_SKIP_PIP:-0}" != "1" ]]; then
  if APRIL_HOME="$REPO_ROOT" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    .venv/bin/python -c "import apps.runner.main" >/dev/null 2>&1; then
    echo "APRIL is already importable from .venv; skipping editable reinstall."
  else
    .venv/bin/pip install -e ".[dev]"
  fi
fi

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

ARGS=(--install --repo-root "$REPO_ROOT" --bin-dir "$BIN_DIR")
if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi
if [[ "$ADD_TO_PATH" == "1" ]]; then
  ARGS+=(--add-to-path)
fi

.venv/bin/python -m apps.runner.install "${ARGS[@]}"
