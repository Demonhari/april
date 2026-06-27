#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
INSTALL_EXTRAS="dev"
INSTALL_GLOBAL=0
ADD_TO_PATH=0

show_help() {
  cat <<'HELP'
Usage: scripts/setup_mac.sh [--runtime] [--voice] [--base] [--global] [--add-to-path]

Creates a local .venv and installs APRIL. This script never runs Homebrew,
never uses sudo, and never downloads model files.

Options:
  --runtime      Include the optional llama.cpp runtime extra.
  --voice        Include optional local voice dependencies.
  --base         Install only base APRIL dependencies, without the dev extra.
  --global       Install the global run/april-run wrappers after setup.
  --add-to-path  With --global, add ~/.local/bin to zsh/bash config.
HELP
}

for arg in "$@"; do
  case "$arg" in
    --runtime) INSTALL_EXTRAS="${INSTALL_EXTRAS},runtime" ;;
    --voice) INSTALL_EXTRAS="${INSTALL_EXTRAS},voice" ;;
    --base) INSTALL_EXTRAS="" ;;
    --global) INSTALL_GLOBAL=1 ;;
    --add-to-path) ADD_TO_PATH=1 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$ADD_TO_PATH" == "1" && "$INSTALL_GLOBAL" != "1" ]]; then
  echo "--add-to-path is only valid with --global" >&2
  exit 2
fi

echo "Detected architecture: $(uname -m)"
cd "$ROOT_DIR"
if [[ "${APRIL_SETUP_SKIP_PIP:-0}" != "1" ]]; then
  "$PYTHON_BIN" -m venv .venv
  if [[ -n "$INSTALL_EXTRAS" ]]; then
    .venv/bin/pip install -e ".[${INSTALL_EXTRAS}]" -c constraints-dev.txt
  else
    .venv/bin/pip install -e . -c constraints-dev.txt
  fi
fi

if [[ "$INSTALL_GLOBAL" == "1" ]]; then
  INSTALL_ARGS=()
  if [[ "$ADD_TO_PATH" == "1" ]]; then
    INSTALL_ARGS+=(--add-to-path)
  fi
  APRIL_INSTALL_SKIP_PIP=1 bash scripts/install_run_april.sh "${INSTALL_ARGS[@]}"
fi

echo "APRIL setup complete."
if [[ "$INSTALL_GLOBAL" == "1" ]]; then
  if [[ "$ADD_TO_PATH" == "1" ]]; then
    echo "Reload your shell, then run: run april --fake"
  else
    echo 'If run is not found, use: export PATH="$HOME/.local/bin:$PATH"'
    echo 'Then run: run april --fake'
  fi
fi
