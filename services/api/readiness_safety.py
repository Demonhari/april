from __future__ import annotations

from pathlib import Path
from typing import Any

from april_common.settings import (
    INSECURE_API_TOKENS,
    INSECURE_RUNTIME_TOKENS,
    AprilSettings,
)
from services.api.reporting import _basename, _redact_path_text


def redact_health_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key in {"path", "model_path", "binary_path"}
                else redact_health_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_health_payload(item) for item in value]
    return value


def safe_runtime_health(payload: dict[str, Any]) -> dict[str, Any]:
    safe = redact_health_payload(payload)
    if isinstance(safe, dict) and isinstance(safe.get("models"), list):
        backend = str(safe.get("backend", "unknown"))
        safe["models"] = [
            safe_model_entry(model, backend) for model in safe["models"] if isinstance(model, dict)
        ]
    return safe if isinstance(safe, dict) else {"status": "unknown"}


def safe_model_entry(model: dict[str, Any], runtime_backend: str) -> dict[str, Any]:
    path = model.get("path")
    backend = str(model.get("backend") or runtime_backend or "unknown")
    return {
        "id": str(model.get("id", "unknown")),
        "name": str(model.get("name", "unknown")),
        "role": str(model.get("role", "unknown")),
        "backend": backend,
        "state": str(model.get("state", "unknown")),
        "keep_loaded": bool(model.get("keep_loaded", False)),
        "missing_path": bool(model.get("missing_path", False)),
        "simulated": backend == "fake" or runtime_backend == "fake",
        "path_basename": _basename(path),
        "context_size": model.get("context_size"),
        "load_error": (
            _redact_path_text(str(model.get("load_error"))) if model.get("load_error") else None
        ),
    }


def voice_artifact(settings: AprilSettings, name: str, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"name": name, "configured": False, "missing": True, "basename": None}
    resolved = settings.resolve_path(path)
    return {
        "name": name,
        "configured": True,
        "missing": not resolved.exists(),
        "basename": resolved.name,
    }


def development_token_warning(settings: AprilSettings) -> str | None:
    if not settings.api.token or settings.api.token in INSECURE_API_TOKENS:
        return "API token uses an insecure development/placeholder default or is empty."
    if not settings.runtime.token or settings.runtime.token in INSECURE_RUNTIME_TOKENS:
        return "Runtime token uses an insecure development/placeholder default or is missing."
    return None
