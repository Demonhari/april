from __future__ import annotations

import json
from typing import Any

from april_common.benchmark_evidence import (
    empty_benchmark_evidence,
    evaluate_benchmark_evidence,
)
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.hardware_profile import safe_hardware_profile
from april_common.settings import AprilSettings
from services.memory.database import Database


async def latest_benchmark_evidence(
    database: Database,
    settings: AprilSettings,
) -> dict[str, object]:
    try:
        row = await database.fetchone(
            """
            SELECT result_json FROM background_jobs
            WHERE job_type = 'model_setup_comparison' AND status = 'succeeded'
            ORDER BY completed_at DESC LIMIT 1
            """
        )
    except Exception:
        return empty_benchmark_evidence()
    if row is None:
        return empty_benchmark_evidence()
    try:
        payload = json.loads(str(row["result_json"]))
    except (KeyError, TypeError, json.JSONDecodeError):
        return {**empty_benchmark_evidence(), "incomplete": True}
    return evaluate_benchmark_evidence(
        payload,
        current_hardware_id=safe_hardware_profile()["id"],
        current_config_fingerprint=config_fingerprint_digest(settings.home),
    )


def api_evidence_boundaries(
    *,
    settings: AprilSettings,
    model_registry: dict[str, Any],
    verified_models: set[str],
    embeddings: dict[str, Any],
    live_flags: dict[str, bool],
    benchmark_evidence: dict[str, object],
    finetuning_status: str,
    rollout_state: dict[str, Any],
) -> dict[str, str]:
    required_models = set(model_registry.get("required_model_ids") or [])
    reasoning_models = set(model_registry.get("reasoning_model_ids") or [])
    return {
        "readiness_implementation": "implemented_in_code",
        "core_models": (
            "verified_with_real_evidence"
            if required_models and required_models.issubset(verified_models)
            else (
                "preflight_ready"
                if model_registry.get("production_model_artifacts_ready")
                else "blocked_for_safety"
            )
        ),
        "reasoning_role": (
            "verified_with_real_evidence"
            if reasoning_models and reasoning_models.issubset(verified_models)
            else ("configured" if reasoning_models else "blocked_for_safety")
        ),
        "semantic_embeddings": (
            "verified_with_real_evidence"
            if embeddings["active_provider"] == "runtime-local"
            and not embeddings["reindex_required"]
            and not embeddings["fell_back_to_hashed_token"]
            else "blocked_for_safety"
        ),
        "push_to_talk": (
            "verified_with_real_evidence"
            if live_flags["voice_live_verified"]
            else ("configured" if settings.voice.enabled else "optional_unavailable")
        ),
        "wake_word": (
            "verified_with_real_evidence"
            if live_flags["wake_word_live_verified"]
            else ("configured" if settings.wake.enabled else "optional_unavailable")
        ),
        "voice_conversation": (
            "verified_with_real_evidence"
            if live_flags["voice_conversation_live_verified"]
            else ("configured" if settings.voice.enabled else "optional_unavailable")
        ),
        "speaker_verification": (
            "verified_with_real_evidence"
            if live_flags["speaker_live_verified"]
            else (
                "blocked_for_safety"
                if settings.wake.speaker_gate == "soft"
                else "optional_unavailable"
            )
        ),
        "model_comparison": (
            "verified_with_real_evidence"
            if benchmark_evidence["production_eligible"]
            else "optional_unavailable"
        ),
        "fine_tuning": (
            "preflight_ready" if finetuning_status == "ready" else "optional_unavailable"
        ),
        "evolution": (
            "preflight_ready"
            if settings.evolution.enabled and rollout_state.get("status") != "degraded"
            else (
                "blocked_for_safety"
                if rollout_state.get("status") == "degraded"
                else "optional_unavailable"
            )
        ),
        "lora_canary": (
            "configured"
            if rollout_state.get("lora_canary_supported")
            and rollout_state.get("status") != "degraded"
            else "blocked_for_safety"
        ),
        "apple_release": "optional_unavailable",
    }
