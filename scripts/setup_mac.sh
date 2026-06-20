#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
INSTALL_EXTRAS="dev"

show_help() {
  cat <<'HELP'
Usage: scripts/setup_mac.sh [--runtime] [--voice] [--base]

Creates a local .venv and installs APRIL. This script never runs Homebrew,
never uses sudo, and never downloads model files.
HELP
}

for arg in "$@"; do
  case "$arg" in
    --runtime) INSTALL_EXTRAS="${INSTALL_EXTRAS},runtime" ;;
    --voice) INSTALL_EXTRAS="${INSTALL_EXTRAS},voice" ;;
    --base) INSTALL_EXTRAS="" ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "Detected architecture: $(uname -m)"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m venv .venv
if [[ -n "$INSTALL_EXTRAS" ]]; then
  .venv/bin/pip install -e ".[${INSTALL_EXTRAS}]"
else
  .venv/bin/pip install -e .
fi

echo "APRIL setup complete."
