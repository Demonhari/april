"""Validated, atomic JSON report output for runner commands."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any


class ReportPathError(ValueError):
    """The operator-supplied report path cannot be used safely."""


def preflight_report_path(path: Path) -> Path:
    """Validate a report destination before loading models or starting services."""

    target = path.expanduser()
    if target.exists() and target.is_dir():
        raise ReportPathError(f"report path is an existing directory: {target}")
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReportPathError(f"cannot create report parent for {target}: {exc}") from exc
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ReportPathError(f"report parent is not writable: {parent}")
    if target.exists() and not os.access(target, os.W_OK):
        raise ReportPathError(f"report file is not writable: {target}")
    return target


def write_json_report(path: Path, payload: Any) -> Path:
    """Serialize and atomically replace a JSON report beside its target."""

    target = preflight_report_path(path)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()
    return target
