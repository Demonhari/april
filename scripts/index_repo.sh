#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$ROOT_DIR}"
cd "$ROOT_DIR"
.venv/bin/python - <<'PY' "$TARGET"
import asyncio
import sys

from skills.code.repo_indexer import repo_indexer


async def main() -> None:
    result = await repo_indexer({"repo_path": sys.argv[1]})
    print(result.stdout)
    if not result.ok:
        raise SystemExit(1)


asyncio.run(main())
PY
