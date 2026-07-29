#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required for the explicit clean locked-install check.\n' >&2
  exit 2
fi

scratch="$(mktemp -d "${TMPDIR:-/tmp}/april-locked-install.XXXXXX")"
cleanup() {
  rm -rf "$scratch"
}
trap cleanup EXIT

cp "$repo_root/pyproject.toml" "$scratch/pyproject.toml"
cp "$repo_root/uv.lock" "$scratch/uv.lock"
cp "$repo_root/README.md" "$scratch/README.md"

uv sync \
  --directory "$scratch" \
  --locked \
  --extra dev \
  --extra security \
  --no-install-project

PYTHONPATH="$repo_root" "$scratch/.venv/bin/python" -c \
  "from services.memory.encryption import AESGCMEncryption; c=AESGCMEncryption(); k=b'k'*32; n=b'n'*12; a=b'id'; assert c.decrypt(k,n,c.encrypt(k,n,b'local',a),a)==b'local'"

printf 'Clean locked base/development/security dependency installation succeeded.\n'
