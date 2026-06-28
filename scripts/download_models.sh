#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
PYTHON="${APRIL_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

show_help() {
  cat <<'HELP'
Usage:
  scripts/download_models.sh --all-core --apply --yes [--skip-existing]
  scripts/download_models.sh --role brain --apply --yes [--skip-existing]

Manifest mode delegates to:
  run april model download ...

APRIL never downloads models during tests, CI, config validation, import, or
fake verification. This script downloads only when you run it explicitly.

Legacy explicit URL mode is still available but deprecated:
  scripts/download_models.sh --url URL --out FILENAME [--sha256 HASH] [--overwrite]
HELP
}

legacy_download() {
  local overwrite="false"
  local sha256=""
  local url=""
  local out=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --url) url="$2"; shift 2 ;;
      --out) out="$2"; shift 2 ;;
      --sha256) sha256="$2"; shift 2 ;;
      --overwrite|--force) overwrite="true"; shift ;;
      -h|--help) show_help; exit 0 ;;
      *) echo "Unknown legacy option: $1" >&2; exit 2 ;;
    esac
  done

  if [[ -z "$url" || -z "$out" ]]; then
    show_help
    exit 2
  fi

  mkdir -p "$MODEL_DIR"
  local target="$MODEL_DIR/$out"
  local part="$target.part"
  if [[ -e "$target" && "$overwrite" != "true" ]]; then
    echo "Refusing to overwrite existing file: $target" >&2
    exit 1
  fi
  if [[ -e "$part" ]]; then
    echo "Refusing to overwrite existing partial file: $part" >&2
    exit 1
  fi

  echo "Deprecated explicit URL mode; prefer: run april model download --all-core --apply --yes" >&2
  curl -L --fail "$url" -o "$part"
  if [[ -n "$sha256" ]]; then
    local actual
    actual="$(shasum -a 256 "$part" | awk '{print $1}')"
    if [[ "$actual" != "$sha256" ]]; then
      echo "SHA-256 mismatch. Expected $sha256 got $actual" >&2
      rm -f "$part"
      exit 1
    fi
  fi
  mv -f "$part" "$target"
  echo "Saved $target"
}

if [[ $# -eq 0 ]]; then
  show_help
  exit 2
fi

for arg in "$@"; do
  if [[ "$arg" == "--url" || "$arg" == "--out" ]]; then
    legacy_download "$@"
    exit $?
  fi
done

exec "$PYTHON" -m apps.runner.main april model download "$@"
