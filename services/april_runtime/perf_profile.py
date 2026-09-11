"""Safe loading and application of operator-created runtime profiles."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from april_common.hardware_profile import cpu_brand, physical_cpu_count
from services.april_runtime.model_registry import ModelDefinition

TUNABLE_FIELDS = frozenset({"threads", "threads_batch", "n_batch", "n_ubatch", "flash_attn"})


def llama_cpp_version() -> str | None:
    try:
        return version("llama-cpp-python")
    except PackageNotFoundError:  # pragma: no cover - optional package metadata
        return None


def model_identity(model: ModelDefinition, root: Path) -> dict[str, object] | None:
    path = model.resolved_path(root)
    try:
        stat = path.stat()
    except OSError:  # pragma: no cover - filesystem race/error path
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def profile_inputs(model: ModelDefinition, root: Path) -> dict[str, object]:
    return {
        "model_id": model.id,
        "model_identity": model_identity(model, root),
        "adapter_identity": _adapter_identity(model, root),
        "llama_cpp_python": llama_cpp_version(),
        "cpu_brand": cpu_brand(),
        "physical_cores": physical_cpu_count(),
        "logical_cores": os.cpu_count(),
        "context_size": model.context_size,
        "chat_format": model.chat_format,
    }


def profile_fingerprint(inputs: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def apply_tunable_overlay(model: ModelDefinition, profile: dict[str, Any]) -> ModelDefinition:
    """Apply only explicitly tunable values from a validated profile."""
    values = profile.get("settings", profile)
    if not isinstance(values, dict):
        return model
    update = {key: values[key] for key in TUNABLE_FIELDS if key in values}
    return model.model_copy(update=update)


def load_matching_profile(root: Path, model: ModelDefinition) -> ModelDefinition:
    """Return an overlay only when a profile's immutable inputs still match."""
    directory = root / "data" / "perf" / "profiles"
    current = profile_inputs(model, root)
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "april.perf.profile.v2"
            or payload.get("model_id") != model.id
        ):
            continue
        stored = payload.get("fingerprint_inputs")
        if stored != current:
            continue
        if payload.get("fingerprint") != profile_fingerprint(current):  # pragma: no cover
            continue
        return apply_tunable_overlay(model, payload)
    return model


async def load_matching_profile_async(root: Path, model: ModelDefinition) -> ModelDefinition:
    return await asyncio.to_thread(load_matching_profile, root, model)


def profile_status(root: Path, models: list[ModelDefinition], mode: str) -> str:
    if mode != "auto":
        return "off"
    directory = root / "data" / "perf" / "profiles"
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:  # pragma: no cover - filesystem error path
        return "stale"
    if not paths:  # pragma: no cover - exercised by operator profile directories
        return "none"
    for model in models:
        current = profile_inputs(model, root)
        fingerprint = profile_fingerprint(current)
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                isinstance(payload, dict)
                and payload.get("schema") == "april.perf.profile.v2"
                and payload.get("model_id") == model.id
                and payload.get("fingerprint") == fingerprint
                and payload.get("fingerprint_inputs") == current
            ):
                return "active"
    return "stale"


def _adapter_identity(model: ModelDefinition, root: Path) -> dict[str, object] | None:
    path = model.resolved_adapter_path(root)
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:  # pragma: no cover - filesystem race/error path
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
