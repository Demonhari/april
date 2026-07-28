from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def adapter_pointer_path(root: Path, model_id: str) -> Path:
    _validate_model_id(model_id)
    return root.resolve() / "data" / "evolution" / "adapters" / f"{model_id}.json"


def read_adapter_pointer(root: Path, model_id: str) -> dict[str, Any] | None:
    path = adapter_pointer_path(root, model_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter pointer for {model_id} must be a JSON object.")
    versions = payload.get("versions")
    if not isinstance(versions, list):
        raise ValueError(f"Adapter pointer for {model_id} must contain a versions list.")
    return payload


def active_adapter_path_from_pointer(root: Path, model_id: str) -> Path | None:
    pointer = read_adapter_pointer(root, model_id)
    if pointer is None:
        return None
    active = _effective_active_version(pointer)
    if active is None:
        return None
    entry = _pointer_entry(pointer, active)
    if entry is None:
        raise ValueError(f"Adapter pointer for {model_id} references an unknown active version.")
    raw_path = Path(str(entry.get("adapter_path") or ""))
    if not str(raw_path):
        raise ValueError(f"Adapter pointer for {model_id} has an empty adapter path.")
    expanded = raw_path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (root / expanded).resolve(strict=False)


def _effective_active_version(pointer: dict[str, Any]) -> int | None:
    """Return only a committed adapter version.

    New lifecycle operations publish a recoverable pending pointer before the
    matching SQLite state is committed. Runtime deliberately continues using
    the previous version during that window.
    """

    operation = pointer.get("pending_operation")
    if isinstance(operation, dict):
        previous = operation.get("previous_active_version")
        return int(previous) if previous is not None else None
    return _pointer_active_version(pointer)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pointer_active_version(pointer: dict[str, Any]) -> int | None:
    value = pointer.get("active_version")
    if value is None:
        return None
    return int(value)


def _pointer_entry(pointer: dict[str, Any], version: int) -> dict[str, Any] | None:
    versions = pointer.get("versions")
    if not isinstance(versions, list):
        return None
    for item in versions:
        if isinstance(item, dict) and int(item.get("version", 0)) == version:
            return item
    return None


def _validate_model_id(model_id: str) -> None:
    if not _MODEL_ID_RE.fullmatch(model_id):
        raise ValueError("model_id is not safe for an adapter pointer filename")
