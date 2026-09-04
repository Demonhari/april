#!/usr/bin/env python3
"""Reject tracked local runtime state and other non-source artifacts."""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent


def forbidden_reason(path: str) -> str | None:
    """Return a safe reason code for a tracked path, without opening it."""
    normalized = path.replace("\\", "/")
    name = PurePosixPath(normalized).name
    if normalized.startswith("data/runtime/"):
        return "runtime_state"
    if normalized.startswith("data/setup/"):
        return "setup_state"
    if fnmatch.fnmatch(normalized, "data/*.write.lock"):
        return "database_write_lock"
    if fnmatch.fnmatch(normalized, "configs/*.bak-*"):
        return "source_config_backup"
    if normalized.startswith("data/verification/"):
        return "generated_verification_report"
    if name in {"capability", "request-outcomes.json"} and normalized.startswith("data/"):
        return "runtime_worker_artifact"
    if name in {".env", ".env.local", ".env.production"}:
        return "credential_file"
    if normalized.startswith("data/") and PurePosixPath(normalized).suffix.lower() in {
        ".db",
        ".sqlite",
        ".sqlite3",
    }:
        return "database"
    if PurePosixPath(normalized).suffix.lower() in {".gguf", ".safetensors", ".onnx"}:
        return "model_or_voice_binary"
    if normalized.startswith("models/") and name.endswith(".bin"):
        return "model_binary"
    if normalized.startswith("data/") and any(
        token in name.casefold() for token in ("credential", "secret", "token")
    ):
        return "credential_or_secret"
    return None


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> int:
    violations = [(path, forbidden_reason(path)) for path in tracked_paths()]
    violations = [(path, reason) for path, reason in violations if reason is not None]
    if violations:
        print(
            f"Source hygiene failed: {len(violations)} tracked local artifact(s).",
            file=sys.stderr,
        )
        for path, reason in violations:
            print(f"- {reason}: {path}", file=sys.stderr)
        return 1
    print("Source hygiene passed: no forbidden tracked local artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
