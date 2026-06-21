#!/usr/bin/env bash
set -euo pipefail

APRIL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_PROJECT="${APRIL_SMOKE_PROJECT:-/tmp/april-smoke-project}"

rm -rf "$SMOKE_PROJECT"
mkdir -p "$SMOKE_PROJECT"
cd "$SMOKE_PROJECT"
git init
git config user.email "april-smoke@example.local"
git config user.name "APRIL Smoke"
printf '# smoke\n' > README.md
git add README.md
git commit -m "initial"

cd "$APRIL_REPO"
export APRIL_ALLOWED_FILESYSTEM_ROOTS="$SMOKE_PROJECT"
run april stop >/dev/null 2>&1 || true

PROJECT_OUTPUT="$(run april project add "$SMOKE_PROJECT" --fake)"
PROJECT_ID="$(
  PROJECT_OUTPUT="$PROJECT_OUTPUT" python3 - <<'PY'
import ast
import os
import re

text = os.environ["PROJECT_OUTPUT"]
match = re.search(r"\{.*\}", text, re.S)
if not match:
    raise SystemExit("project add output did not contain a dict")
data = ast.literal_eval(match.group(0))
print(data["id"])
PY
)"

run april agent run coding_agent "Inspect this repository" --project-id "$PROJECT_ID" --fake
