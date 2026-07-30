"""Fail-closed readers for redacted real-model verification evidence."""

from __future__ import annotations

import json
from pathlib import Path

from april_common.config_fingerprint import config_fingerprint_digest
from april_common.report_freshness import freshness_from_payload


def verified_model_ids(home: Path) -> set[str]:
    path = home / "data" / "verification" / "mac-readiness.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    if (
        payload.get("report_type") != "multi_model"
        or payload.get("runtime_backend") != "llama_cpp"
        or payload.get("real_model_verified") is not True
        or payload.get("all_configured_models_verified") is not True
        or payload.get("verification_level") != "all"
    ):
        return set()
    freshness = freshness_from_payload(
        payload,
        report_type="multi_model",
        current_fingerprint=config_fingerprint_digest(home),
        basename=path.name,
    )
    models = payload.get("models")
    if freshness.stale or not isinstance(models, list):
        return set()
    return {
        str(model["model_id"])
        for model in models
        if isinstance(model, dict)
        and model.get("available") is True
        and model.get("load_success") is True
        and model.get("chat_success") is True
        and model.get("streaming_success") is True
        and model.get("unload_success") is True
        and (
            model.get("structured_brain_json_success") is True
            if model.get("role") == "brain"
            else (
                model.get("smoke_success") is True and model.get("smoke_schema_valid") is not False
            )
        )
        and isinstance(model.get("model_id"), str)
    }
