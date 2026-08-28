#!/usr/bin/env python3
"""Select and validate an APRIL-supported Python interpreter."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

MIN_SUPPORTED = (3, 11)
MAX_SUPPORTED = (3, 13)


def _version(executable: str) -> tuple[int, int, str] | None:
    try:
        completed = subprocess.run(
            [executable, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    fields = completed.stdout.strip().split()
    if len(fields) != 2:
        return None
    try:
        major, minor = int(fields[0]), int(fields[1])
    except ValueError:
        return None
    return major, minor, f"Python {major}.{minor}"


def _resolve(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    return shutil.which(value)


def select(explicit: str | None = None) -> tuple[str, str]:
    requested = explicit or os.environ.get("PYTHON_BIN") or os.environ.get("PYTHON")
    candidates = [requested] if requested else ["python3.13", "python3.12", "python3.11", "python3"]
    rejected: list[str] = []
    for value in candidates:
        if value is None:
            continue
        resolved = _resolve(value)
        if resolved is None:
            if requested:
                raise RuntimeError(
                    f"Requested Python interpreter does not exist or is not executable: {value}"
                )
            continue
        detected = _version(resolved)
        if detected is None:
            if requested:
                raise RuntimeError(f"Could not execute requested Python interpreter: {value}")
            continue
        major, minor, label = detected
        if (major, minor) < MIN_SUPPORTED or (major, minor) > MAX_SUPPORTED:
            rejected.append(f"{resolved} ({label})")
            if requested:
                raise RuntimeError(
                    f"Unsupported Python version {label}; APRIL requires Python 3.11, "
                    "3.12, or 3.13."
                )
            continue
        return resolved, label
    detail = f" Rejected: {', '.join(rejected)}." if rejected else ""
    raise RuntimeError(
        "No supported Python interpreter was found; install Python 3.11, 3.12, or 3.13 "
        "and set PYTHON_BIN (or PYTHON) to its executable." + detail
    )


def main() -> int:
    if any(arg in sys.argv[1:] for arg in ("--help", "-h")):
        print("Usage: select_python.py [--path-only] [INTERPRETER]")
        print("Accepts Python 3.11, 3.12, or 3.13; honors PYTHON_BIN and PYTHON.")
        return 0
    path_only = "--path-only" in sys.argv[1:]
    explicit_values = [arg for arg in sys.argv[1:] if arg != "--path-only"]
    if len(explicit_values) > 1:
        print("usage: select_python.py [--path-only] [INTERPRETER]", file=sys.stderr)
        return 2
    try:
        executable, label = select(explicit_values[0] if explicit_values else None)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(executable if path_only else f"Selected APRIL Python: {executable} ({label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
