#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
OVERWRITE="false"
SHA256=""
URL=""
OUT=""

show_help() {
  cat <<'HELP'
Usage: scripts/download_models.sh --url URL --out FILENAME [--sha256 HASH] [--overwrite]

APRIL does not provide undocumented model URLs and does not download models
automatically. You must supply an explicit URL or manually place GGUF files in
models/. Existing files are never overwritten unless --overwrite is supplied.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --sha256) SHA256="$2"; shift 2 ;;
    --overwrite) OVERWRITE="true"; shift ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$URL" || -z "$OUT" ]]; then
  show_help
  exit 2
fi

mkdir -p "$MODEL_DIR"
TARGET="$MODEL_DIR/$OUT"
if [[ -e "$TARGET" && "$OVERWRITE" != "true" ]]; then
  echo "Refusing to overwrite existing file: $TARGET" >&2
  exit 1
fi

curl -L --fail "$URL" -o "$TARGET"
if [[ -n "$SHA256" ]]; then
  ACTUAL="$(shasum -a 256 "$TARGET" | awk '{print $1}')"
  if [[ "$ACTUAL" != "$SHA256" ]]; then
    echo "SHA-256 mismatch. Expected $SHA256 got $ACTUAL" >&2
    rm -f "$TARGET"
    exit 1
  fi
fi

echo "Saved $TARGET"
