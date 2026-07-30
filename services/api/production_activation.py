"""Production-only activation gates for the Core API readiness surface."""

from __future__ import annotations

import importlib.util
from typing import Any

from april_common.service_health import ServiceHealthResult
from april_common.settings import AprilSettings


def finetuning_readiness(settings: AprilSettings) -> dict[str, Any]:
    if not settings.finetune.enabled:
        return {
            "status": "disabled",
            "enabled": False,
            "trainer_configured": False,
            "evaluator_configured": False,
            "action": "run april finetune doctor",
        }
    trainer = settings.finetune.trainer_executable
    evaluator = settings.finetune.evaluator_executable
    trainer_ready = bool(trainer and settings.resolve_path(trainer).is_file())
    evaluator_ready = bool(evaluator and settings.resolve_path(evaluator).is_file())
    return {
        "status": "ready" if trainer_ready and evaluator_ready else "unconfigured",
        "enabled": True,
        "trainer_configured": trainer_ready,
        "evaluator_configured": evaluator_ready,
        "action": None if trainer_ready and evaluator_ready else "run april finetune doctor",
    }


def production_activation_failure_reasons(
    *,
    settings: AprilSettings,
    runtime_probe: ServiceHealthResult,
    runtime_backend: str,
    runtime_simulated: bool | None,
    model_registry: dict[str, Any],
    verified_models: set[str],
    embeddings: dict[str, Any],
    live_flags: dict[str, bool],
    job_worker_ready: bool,
    tool_worker_protocol_ready: bool,
    tool_worker_self_check: bool,
    audit_chain_status: str,
    database_integrity: Any,
    rollout_state: dict[str, Any],
    finetuning: dict[str, Any],
    credential_store_selected: str,
    legacy_plaintext_credential_detected: bool,
) -> list[dict[str, str]]:
    """Return stable production blockers without changing operational readiness."""
    reasons: list[dict[str, str]] = []

    def add(code: str, message: str, action: str) -> None:
        reasons.append({"code": code, "message": message, "action": action})

    if not runtime_probe.ok:
        add(
            f"runtime_{runtime_probe.reason}",
            "Authenticated Runtime liveness did not pass.",
            "run april doctor --daily-driver",
        )
    if runtime_backend != "llama_cpp" or runtime_simulated is not False:
        add(
            "runtime_simulated",
            "A fake, simulated, or unspecified Runtime is not production evidence.",
            "run april verify --all-configured-models --require-real-model "
            "--report data/verification/mac-readiness.json",
        )
    if importlib.util.find_spec("llama_cpp") is None:
        add(
            "llama_cpp_python_unavailable",
            "llama-cpp-python is unavailable.",
            "pip install -e '.[runtime]'",
        )
    if not bool(model_registry.get("production_model_artifacts_ready")):
        add(
            "required_model_artifacts_unavailable",
            "Brain, coding, and reading registrations must name readable GGUF files.",
            "run april setup models",
        )
    required_ids = set(model_registry.get("required_model_ids") or [])
    if not required_ids or not required_ids.issubset(verified_models):
        add(
            "real_model_evidence_missing",
            "Required Brain, coding, and reading models lack fresh real verification.",
            "run april verify --all-configured-models --require-real-model "
            "--report data/verification/mac-readiness.json",
        )
    reasoning_ids = set(model_registry.get("reasoning_model_ids") or [])
    if not reasoning_ids:
        add(
            "reasoning_role_unregistered",
            "No reasoning-role model is registered and genuinely verified.",
            "run april model import --role reasoning --id qwen3-4b-reasoning "
            '--name "Qwen3-4B Q4_K_M" --path /ABSOLUTE/LOCAL/PATH '
            "--sha256 EXPECTED_SHA256",
        )
    elif not reasoning_ids.issubset(verified_models):
        add(
            "reasoning_role_unverified",
            "The reasoning-role model lacks fresh real verification.",
            "run april verify --all-configured-models --require-real-model "
            "--report data/verification/mac-readiness.json",
        )
    if embeddings.get("active_provider") != "runtime-local":
        add(
            "hashed_token_embeddings_active",
            "Runtime-local semantic embeddings are not active.",
            "run april model import --role embedding --id nomic-embed-text-v1.5 "
            '--name "nomic-embed-text-v1.5 Q8" --path /ABSOLUTE/LOCAL/PATH '
            "--sha256 EXPECTED_SHA256",
        )
    embedding_model_id = embeddings.get("embedding_model_id")
    if embeddings.get("active_provider") == "runtime-local" and (
        not isinstance(embedding_model_id, str) or embedding_model_id not in verified_models
    ):
        add(
            "embedding_model_unverified",
            "The active embedding model lacks fresh real-model verification.",
            "run april verify --all-configured-models --require-real-model "
            "--report data/verification/mac-readiness.json",
        )
    if embeddings.get("fell_back_to_hashed_token"):
        add(
            "runtime_embedding_fallback",
            "Runtime-local embeddings fell back to hashed-token.",
            "run april memory reindex --wait",
        )
    if embeddings.get("reindex_required") or not embeddings.get("active_generation"):
        add(
            "semantic_reindex_required",
            "A compatible active semantic vector generation is required.",
            "run april memory reindex --wait",
        )
    if settings.voice.enabled and not live_flags["voice_conversation_live_verified"]:
        add(
            "live_voice_not_verified",
            "Voice is configured but complete real-hardware verification is absent.",
            "run april voice verify-conversation-live "
            "--report data/verification/voice-conversation-live.json",
        )
    if settings.workers.job_worker_enabled and not job_worker_ready:
        add("job_worker_unavailable", "The durable Job Worker is not ready.", "run april status")
    if settings.workers.tool_worker_enabled and (
        not tool_worker_protocol_ready or not tool_worker_self_check
    ):
        add(
            "tool_worker_unavailable",
            "The Tool Worker protocol or sandbox self-check is not ready.",
            "run april status",
        )
    if finetuning["status"] != "ready":
        add(
            f"fine_tuning_{finetuning['status']}",
            "Fine-tuning is disabled or its trainer/evaluator is unconfigured.",
            "run april finetune doctor",
        )
    if not settings.evolution.enabled:
        add(
            "evolution_disabled",
            "Evolution is intentionally disabled.",
            "Review the evolution operator documentation; do not enable it automatically.",
        )
    elif rollout_state.get("status") == "degraded":
        add(
            "evolution_unsafe_transition",
            "Evolution has an unsafe incomplete transition.",
            str(rollout_state.get("action") or "run april evolve rollout list"),
        )
    if audit_chain_status not in {"valid", "anchor_lagged"}:
        add("audit_chain_unverified", "The audit chain is not verified.", "run april audit verify")
    if not database_integrity.ok:
        add(
            "database_integrity_failed",
            "SQLite integrity, foreign keys, WAL settings, and migrations must all pass.",
            "run april database check",
        )
    if settings.environment == "production":
        if not settings.api.token or not settings.runtime.token:
            add(
                "production_credentials_unavailable",
                "Production API and Runtime credentials are unavailable.",
                "run april setup tokens",
            )
        if credential_store_selected != "macos-keychain":
            add(
                "production_keychain_unavailable",
                "Production credentials are not sourced from macOS Keychain.",
                "run april security credentials migrate",
            )
        if legacy_plaintext_credential_detected:
            add(
                "legacy_plaintext_credentials_detected",
                "Legacy plaintext credentials are present in production mode.",
                "run april security credentials migrate",
            )
    add(
        "apple_release_evidence_unavailable",
        "Signing, notarization, stapling, and Gatekeeper evidence was not evaluated.",
        'run april package sign dist/APRIL.app --identity "Developer ID Application: NAME"',
    )
    return reasons
