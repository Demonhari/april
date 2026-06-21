#!/usr/bin/env bash
set -euo pipefail

FORCE=0
ADD_TO_PATH=0
INSTALL_SHELL="${APRIL_INSTALL_SHELL:-}"

show_help() {
  cat <<'HELP'
Usage: scripts/install_run_april.sh [--force] [--add-to-path] [--shell zsh|bash]

Installs APRIL-owned wrappers into ~/.local/bin. This script never uses sudo
and only edits shell PATH config when --add-to-path is supplied.
HELP
}

while [[ "$#" -gt 0 ]]; do
  arg="$1"
  case "$arg" in
    --force) FORCE=1 ;;
    --add-to-path) ADD_TO_PATH=1 ;;
    --shell)
      if [[ "$#" -lt 2 ]]; then
        echo "Missing value for --shell" >&2
        exit 2
      fi
      INSTALL_SHELL="$2"
      shift
      ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; show_help >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Could not find python3.11 or python3." >&2
  echo "Set PYTHON to a valid Python 3.11 interpreter." >&2
  exit 1
fi

if [[ "${APRIL_INSTALL_SKIP_PIP:-0}" != "1" && ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [[ "${APRIL_INSTALL_SKIP_PIP:-0}" != "1" && ! -x ".venv/bin/python" ]]; then
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

INSTALL_PYTHON="$PYTHON_BIN"
if [[ -x ".venv/bin/python" ]]; then
  INSTALL_PYTHON=".venv/bin/python"
fi

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

ARGS=(--install --repo-root "$REPO_ROOT" --bin-dir "$BIN_DIR")
if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi
if [[ "$ADD_TO_PATH" == "1" ]]; then
  ARGS+=(--add-to-path)
  EXPORTED_SHELL="$(printenv SHELL || true)"
  if [[ -z "$INSTALL_SHELL" && -z "$EXPORTED_SHELL" && "${APRIL_INSTALL_SKIP_PIP:-0}" == "1" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      INSTALL_SHELL="zsh"
    else
      INSTALL_SHELL="bash"
    fi
  fi
  if [[ -n "$INSTALL_SHELL" ]]; then
    ARGS+=(--shell "$INSTALL_SHELL")
  fi
fi

APRIL_HOME="$REPO_ROOT" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$INSTALL_PYTHON" -m apps.runner.install "${ARGS[@]}"
