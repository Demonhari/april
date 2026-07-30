from __future__ import annotations

from collections.abc import Collection

from apps.runner.readiness_models import EvidenceState, ReadinessModel
from april_common.settings import AprilSettings


def build_evidence_boundaries(
    *,
    settings: AprilSettings,
    models: list[ReadinessModel],
    core_model_ids: set[str],
    core_models_verified: bool,
    reasoning_model_ids: set[str],
    verified_model_ids: Collection[str],
    vector_metadata: dict[str, object],
    voice_enabled: bool,
    wake_enabled: bool,
    voice_live_status: str,
    sentinel_live_status: str,
    voice_conversation_live_status: str,
    speaker_live_status: str,
    speaker_soft: bool,
    benchmark_production_eligible: bool,
    fine_tuning_ready: bool,
) -> dict[str, EvidenceState]:
    core_preflight = bool(core_model_ids) and all(
        model.artifact_status == "valid" for model in models if model.id in core_model_ids
    )
    reasoning_verified = bool(reasoning_model_ids) and reasoning_model_ids.issubset(
        verified_model_ids
    )
    semantic_verified = (
        settings.memory.embedding_provider == "runtime-local"
        and vector_metadata.get("provider") == "runtime-local"
        and vector_metadata.get("embedding_model_id") == settings.memory.embedding_model_id
    )
    return {
        "readiness_implementation": "implemented_in_code",
        "core_models": (
            "verified_with_real_evidence"
            if core_models_verified
            else ("preflight_ready" if core_preflight else "blocked_for_safety")
        ),
        "reasoning_role": (
            "verified_with_real_evidence"
            if reasoning_verified
            else ("configured" if reasoning_model_ids else "blocked_for_safety")
        ),
        "semantic_embeddings": (
            "verified_with_real_evidence" if semantic_verified else "blocked_for_safety"
        ),
        "push_to_talk": (
            "verified_with_real_evidence"
            if voice_live_status == "verified"
            else ("configured" if voice_enabled else "optional_unavailable")
        ),
        "wake_word": (
            "verified_with_real_evidence"
            if sentinel_live_status == "verified"
            else ("configured" if wake_enabled else "optional_unavailable")
        ),
        "voice_conversation": (
            "verified_with_real_evidence"
            if voice_conversation_live_status == "verified"
            else ("configured" if voice_enabled else "optional_unavailable")
        ),
        "speaker_verification": (
            "verified_with_real_evidence"
            if speaker_live_status == "verified"
            else ("blocked_for_safety" if speaker_soft else "optional_unavailable")
        ),
        "model_comparison": (
            "verified_with_real_evidence"
            if benchmark_production_eligible
            else "optional_unavailable"
        ),
        "fine_tuning": ("preflight_ready" if fine_tuning_ready else "optional_unavailable"),
        "evolution": ("preflight_ready" if settings.evolution.enabled else "optional_unavailable"),
        "lora_canary": "blocked_for_safety",
        "apple_release": "optional_unavailable",
    }
