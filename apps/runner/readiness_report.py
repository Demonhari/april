from __future__ import annotations

# ruff: noqa: F401
import importlib.util
import json
import platform
import sqlite3
from pathlib import Path
from typing import Any

from apps.runner.mac_report import redact_reason
from apps.runner.readiness_evolution import (
    _pending_eval_case_count,
    _pending_real_runtime_overlay_blockers,
    _pending_write_capable_overlay_count,
)
from apps.runner.readiness_inspection import (
    _active_vector_metadata,
    _active_vector_provider,
    _benchmark_evidence,
    _verified_model_ids,
)
from apps.runner.readiness_model_checks import _build_model_and_registry_checks
from apps.runner.readiness_models import (
    _IMPORT_EMBEDDING,
    _IMPORT_REASONING,
    _INSTALL_RUNTIME,
    _LOOPBACK_HOSTS,
    _SETUP_EMBEDDINGS,
    _SETUP_MODELS,
    _SETUP_TOKENS,
    _SETUP_VOICE,
    _VERIFY_REAL,
    _VERIFY_VOICE,
    _VERIFY_VOICE_CONVERSATION,
    _VERIFY_WAKE,
    CheckStatus,
    ReadinessCheck,
    ReadinessModel,
    ReadinessReport,
    VoiceArtifact,
)
from apps.runner.readiness_security import _token_status
from apps.runner.readiness_voice import (
    _daemon_status,
    _sentinel_live_status,
    _voice_artifact,
    _voice_conversation_live_status,
)
from april_common.audit import audit_logger_for_settings
from april_common.credentials import CredentialStore
from april_common.errors import ConfigError
from april_common.hardware_profile import safe_hardware_profile
from april_common.process_sandbox import SandboxBackend, sandbox_capabilities
from april_common.settings import (
    KNOWN_DEFAULT_API_TOKENS,
    KNOWN_DEFAULT_RUNTIME_TOKENS,
    PLACEHOLDER_API_TOKENS,
    PLACEHOLDER_RUNTIME_TOKENS,
    AprilSettings,
    load_settings,
)
from april_common.time import utc_now_iso
from april_common.token_setup import legacy_plaintext_credentials_detected
from services.april_runtime.model_registry import ModelRegistry
from services.evaluation.model_quality import fixture_set_metadata
from services.evolution.adapters import inspect_adapter_state
from services.evolution.rollouts import inspect_rollout_state
from services.memory.maintenance import check_database


def build_readiness_report(
    home: Path, *, credential_store: CredentialStore | None = None
) -> ReadinessReport:
    root = home.expanduser().resolve()
    checks: list[ReadinessCheck] = []

    try:
        settings = load_settings(root=root, credential_store=credential_store)
    except ConfigError as exc:
        # A broken config blocks everything; report it honestly rather than crash.
        return ReadinessReport(
            generated_at=utc_now_iso(),
            os=f"{platform.system()} {platform.release()}".strip(),
            cpu_architecture=platform.machine(),
            python_version=platform.python_version(),
            runtime_backend="unknown",
            runtime_is_fake=False,
            llama_cpp_python_available=importlib.util.find_spec("llama_cpp") is not None,
            environment="unknown",
            voice_enabled=False,
            checks=[
                ReadinessCheck(
                    name="configuration load",
                    status="blocker",
                    detail=redact_reason(str(exc)),
                    action="run april config validate",
                )
            ],
            blockers=["configuration load"],
            next_actions=["run april config validate"],
        )

    (
        backend,
        runtime_is_fake,
        llama_available,
        models,
        router_model_id,
        router_aliased,
        dedicated_router_available,
        router_failure_reason,
        reading_models,
        reading_available,
        lora_adapter_missing_count,
        adapter_state,
        rollout_state,
        rollout_status,
        embedding_role_models,
        reasoning_role_models,
        vector_metadata,
        fixture_metadata,
        benchmark_evidence,
    ) = _build_model_and_registry_checks(root, settings, checks)

    # --- loopback-only binding ------------------------------------------------
    non_loopback = [
        f"{name}={host}"
        for name, host in (("api.host", settings.api.host), ("runtime.host", settings.runtime.host))
        if host not in _LOOPBACK_HOSTS
    ]
    if non_loopback:
        checks.append(
            ReadinessCheck(
                name="loopback-only binding",
                status="blocker",
                detail="Non-loopback bind address configured: " + ", ".join(sorted(non_loopback)),
                action="Set api.host and runtime.host to 127.0.0.1 (APRIL is loopback-only).",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                name="loopback-only binding",
                status="ok",
                detail="API and runtime bind to loopback only.",
            )
        )

    # --- hashed-token embeddings in production --------------------------------
    if (
        settings.memory.embedding_provider == "hashed-token"
        and settings.environment == "production"
    ):
        checks.append(
            ReadinessCheck(
                name="embedding provider hardening",
                status="warning",
                detail=(
                    "hashed-token embeddings are active in a production environment; "
                    "semantic memory is degraded and hardened go-live holds at warning."
                ),
                action=_SETUP_EMBEDDINGS,
            )
        )
    if (
        settings.environment == "production"
        and settings.memory.embedding_provider != "runtime-local"
    ):
        checks.append(
            ReadinessCheck(
                name="runtime-local embedding hardening",
                status="warning",
                detail=(
                    "Production-like readiness expects memory.embedding_provider=runtime-local; "
                    "hashed-token remains a deterministic degraded fallback."
                ),
                action=_SETUP_EMBEDDINGS,
            )
        )
    if settings.environment == "production" and not embedding_role_models:
        checks.append(
            ReadinessCheck(
                name="embedding-role model registration",
                status="warning",
                detail=(
                    "No role=embedding model is registered; runtime-local semantic memory "
                    "cannot become active until one is added."
                ),
                action=_SETUP_EMBEDDINGS,
            )
        )

    # --- development tokens --------------------------------------------------
    api_status = _token_status(settings.api.token, KNOWN_DEFAULT_API_TOKENS, PLACEHOLDER_API_TOKENS)
    runtime_status = _token_status(
        settings.runtime.token, KNOWN_DEFAULT_RUNTIME_TOKENS, PLACEHOLDER_RUNTIME_TOKENS
    )
    token_statuses = {api_status, runtime_status}
    if "placeholder-insecure" in token_statuses:
        # The .env.example placeholders are not secret. They are fine to discover
        # locally but must be replaced before any non-local exposure; never "ok".
        checks.append(
            ReadinessCheck(
                name="api/runtime tokens",
                status="warning",
                detail="Insecure placeholder tokens from .env.example are still active.",
                action=_SETUP_TOKENS,
            )
        )
    elif "default-development" in token_statuses:
        # Default tokens are fine for local development; they must be rotated
        # before any non-local exposure. A warning, not a hard model blocker.
        checks.append(
            ReadinessCheck(
                name="api/runtime tokens",
                status="warning",
                detail="Default development tokens are still active.",
                action=_SETUP_TOKENS,
            )
        )
    elif "missing" in token_statuses:
        checks.append(
            ReadinessCheck(
                name="api/runtime tokens",
                status="warning",
                detail="A loopback token is not configured.",
                action=_SETUP_TOKENS,
            )
        )
    else:
        checks.append(ReadinessCheck(name="api/runtime tokens", status="ok", detail="configured"))

    credential_store_selected: str = settings.security.credential_store
    if credential_store_selected == "auto":
        credential_store_selected = (
            "macos-keychain"
            if settings.environment == "production" and platform.system() == "Darwin"
            else "legacy-development-default"
        )
    legacy_plaintext = legacy_plaintext_credentials_detected(settings.home)
    checks.append(
        ReadinessCheck(
            name="credential store",
            status="warning" if legacy_plaintext else "ok",
            detail=(
                f"{credential_store_selected}; legacy plaintext credential detected"
                if legacy_plaintext
                else credential_store_selected
            ),
            action=("run april security credentials migrate" if legacy_plaintext else None),
        )
    )
    sandbox = sandbox_capabilities(
        environment=settings.environment,
        development_override=settings.workers.development_unsandboxed_override,
    )
    sandbox_status: CheckStatus = "ok"
    if sandbox.backend is SandboxBackend.UNAVAILABLE:
        sandbox_status = "blocker" if settings.environment == "production" else "warning"
    elif sandbox.development_override_enabled:
        sandbox_status = "warning"
    checks.append(
        ReadinessCheck(
            name="Tool Worker OS sandbox",
            status=sandbox_status,
            detail=(
                sandbox.warning
                or (
                    f"{sandbox.backend.value}; network denial and filesystem policy are OS-enforced"
                )
            ),
            action=(
                "Run APRIL on macOS with /usr/bin/sandbox-exec available."
                if sandbox.backend is SandboxBackend.UNAVAILABLE
                else None
            ),
        )
    )
    try:
        audit_size = settings.audit_path.stat().st_size if settings.audit_path.exists() else 0
        if audit_size <= 4 * 1024 * 1024:
            audit_status = audit_logger_for_settings(settings).verify().status
        else:
            audit_status = "explicit_verification_required"
    except (OSError, RuntimeError):
        audit_status = "unavailable"
    checks.append(
        ReadinessCheck(
            name="audit chain",
            status=("ok" if audit_status in {"valid", "anchor_lagged"} else "warning"),
            detail=audit_status,
            action=(
                "run april audit verify" if audit_status not in {"valid", "anchor_lagged"} else None
            ),
        )
    )
    database_status = check_database(settings.database_path, home=settings.home)
    checks.append(
        ReadinessCheck(
            name="database quick integrity",
            status="ok" if database_status.ok else "warning",
            detail=(
                f"quick_check={database_status.quick_check}; "
                f"foreign_keys={'ok' if database_status.foreign_key_consistent else 'failed'}; "
                f"journal={database_status.journal_mode}; "
                "failures="
                f"{','.join(database_status.failures) if database_status.failures else 'none'}"
            ),
            action="run april database check" if not database_status.ok else None,
        )
    )

    # --- voice artifacts (optional) -----------------------------------------
    voice_enabled = settings.voice.enabled
    voice_specs = (
        ("whisper.cpp binary", settings.voice.whisper_binary_path, True),
        ("whisper model", settings.voice.whisper_model_path, True),
        ("piper binary", settings.voice.piper_binary_path, True),
        ("piper voice model", settings.voice.piper_model_path, True),
    )
    voice_artifacts: list[VoiceArtifact] = []
    for name, voice_path, required in voice_specs:
        artifact, check = _voice_artifact(
            settings, name, voice_path, enabled=voice_enabled, required=required
        )
        voice_artifacts.append(artifact)
        checks.append(check)
    wake_word_paths = settings.voice.effective_wake_word_model_paths
    if wake_word_paths:
        for index, wake_path in enumerate(wake_word_paths):
            name = "wake-word model" if index == 0 else f"wake-word model {index + 1}"
            artifact, check = _voice_artifact(
                settings, name, wake_path, enabled=voice_enabled, required=False
            )
            voice_artifacts.append(artifact)
            checks.append(check)
    else:
        artifact, check = _voice_artifact(
            settings, "wake-word model", None, enabled=voice_enabled, required=False
        )
        voice_artifacts.append(artifact)
        checks.append(check)

    wake_enabled = settings.wake.enabled
    speaker_soft = settings.wake.speaker_gate == "soft"
    speaker_model_path = settings.wake.speaker_verifier_model_path
    speaker_model_configured = speaker_model_path is not None
    speaker_model_exists = bool(
        speaker_model_path is not None and settings.resolve_path(speaker_model_path).is_file()
    )
    if speaker_soft and speaker_model_exists:
        from services.wake.speaker import onnxruntime_importable

        speaker_runtime_available = onnxruntime_importable()
    else:
        speaker_runtime_available = False
    speaker_gate_supported = bool(
        speaker_soft
        and speaker_model_configured
        and speaker_model_exists
        and speaker_runtime_available
    )
    if speaker_gate_supported:
        speaker_detail = (
            "speaker_gate=soft has a configured local ONNX model and ONNX Runtime is "
            "importable. Live scoring still requires target-Mac validation."
        )
    elif speaker_soft and not speaker_model_configured:
        speaker_detail = (
            "speaker_gate=soft is configured without "
            "wake.speaker_verifier_model_path; Sentinel degrades to off with one audited "
            "warning. Follow scripts/speaker_verifier/README.md."
        )
    elif speaker_soft and not speaker_model_exists:
        speaker_detail = (
            "wake.speaker_verifier_model_path does not name an existing local file; "
            "Sentinel degrades to off with one audited warning. Follow "
            "scripts/speaker_verifier/README.md."
        )
    elif speaker_soft:
        speaker_detail = (
            "The optional onnxruntime dependency is not importable; Sentinel degrades "
            "to off with one audited warning. Install APRIL's voice extra and follow "
            "scripts/speaker_verifier/README.md."
        )
    else:
        speaker_detail = (
            "speaker_gate is off. `april voice enroll` records local samples but does "
            "not enable soft mode by itself. Configure wake.speaker_verifier_model_path "
            "as described in scripts/speaker_verifier/README.md before enabling it."
        )
    checks.append(
        ReadinessCheck(
            name="speaker gate",
            status=("ok" if speaker_gate_supported else "warning") if wake_enabled else "skipped",
            detail=(
                speaker_detail
                + " The speaker gate is a convenience filter, never a security boundary."
                + (" Anyone near the microphone can wake APRIL." if wake_enabled else "")
            ),
        )
    )
    voice_conversation_live_status = _voice_conversation_live_status(root)
    checks.append(
        ReadinessCheck(
            name="complete live voice conversation",
            status=(
                "ok"
                if voice_conversation_live_status == "verified"
                else ("warning" if voice_enabled else "skipped")
            ),
            detail=(
                "Two endpointed turns, session continuity, and production barge-in "
                "were verified on real hardware."
                if voice_conversation_live_status == "verified"
                else "Complete two-turn live voice verification has not passed."
            ),
            action=(
                None
                if voice_conversation_live_status == "verified" or not voice_enabled
                else _VERIFY_VOICE_CONVERSATION
            ),
        )
    )
    if wake_enabled and not settings.voice.effective_wake_word_model_paths:
        checks.append(
            ReadinessCheck(
                name="wake-word ONNX model",
                status="blocker",
                detail="wake.enabled is on but no wake-word model path is configured.",
                action=_SETUP_VOICE,
            )
        )
    sentinel_live_status = _sentinel_live_status(root)
    if sentinel_live_status == "verified":
        sentinel_check_status: CheckStatus = "ok"
        sentinel_detail = "Latest wake-word live report verified the Sentinel pipeline."
    elif wake_enabled:
        # Wake is enabled but never live-validated on this machine: warn.
        sentinel_check_status = "warning"
        sentinel_detail = (
            "wake.enabled is on but the Sentinel pipeline has no live validation "
            "record on this Mac."
        )
    else:
        sentinel_check_status = "skipped"
        sentinel_detail = "Sentinel live pipeline has not been verified on this Mac."
    checks.append(
        ReadinessCheck(
            name="Sentinel live verification",
            status=sentinel_check_status,
            detail=sentinel_detail,
            action=(_VERIFY_WAKE if wake_enabled and sentinel_live_status != "verified" else None),
        )
    )

    # --- evolution/scheduler wiring -------------------------------------------
    if settings.evolution.enabled and not settings.scheduler.enabled:
        checks.append(
            ReadinessCheck(
                name="evolution scheduling",
                status="warning",
                detail=(
                    "evolution.enabled is on but scheduler.enabled is off; the nightly "
                    "Dreamer will never run automatically."
                ),
                action="Set scheduler.enabled: true (or run the Dreamer manually).",
            )
        )
    if settings.scheduler.enabled and settings.evolution.enabled:
        kill_switch = settings.evolution_path / "DISABLED"
        if kill_switch.exists():
            checks.append(
                ReadinessCheck(
                    name="evolution kill switch",
                    status="warning",
                    detail="Scheduler and evolution are enabled but the local kill switch "
                    "file is present; Dreamer runs are blocked.",
                    action="Remove data/evolution/DISABLED to re-enable nightly evolution.",
                )
            )

    # --- unreviewed evolution artifacts ---------------------------------------
    pending_overlays = _pending_write_capable_overlay_count(settings)
    if pending_overlays:
        checks.append(
            ReadinessCheck(
                name="prompt overlay review",
                status="warning",
                detail=(
                    f"{pending_overlays} overlay candidate(s) for write-capable agents "
                    "are stored and may await review (already-applied candidates are "
                    "listed precisely by the API)."
                ),
                action="run april evolve overlays pending",
            )
        )
    pending_evals = _pending_eval_case_count(settings)
    if pending_evals:
        checks.append(
            ReadinessCheck(
                name="pending eval cases",
                status="warning",
                detail=(
                    f"{pending_evals} staged eval case(s) have not been reviewed "
                    "(promoted/rejected cases are no longer counted)."
                ),
                action=(
                    "run april evolve evals pending, then promote with "
                    "`april evolve evals promote <case_id> --expected ...` or reject "
                    "with `april evolve evals reject <case_id> --reason ...`"
                ),
            )
        )
    pending_real_runtime_blockers = _pending_real_runtime_overlay_blockers(settings)
    if pending_real_runtime_blockers:
        checks.append(
            ReadinessCheck(
                name="pending real-runtime overlay blockers",
                status="warning",
                detail=(
                    f"{len(pending_real_runtime_blockers)} production overlay candidate(s) "
                    "are held pending because real-runtime eval did not pass."
                ),
                action="run april evolve report",
            )
        )
    production_real_runtime_eval_required = settings.environment == "production"
    checks.append(
        ReadinessCheck(
            name="prompt overlay eval gate",
            status=(
                "warning"
                if production_real_runtime_eval_required and not runtime_is_fake
                else "blocker"
                if production_real_runtime_eval_required
                else "skipped"
            ),
            detail=(
                "Overlay activation requires baseline-versus-candidate behavioral A/B "
                "evaluation. Production additionally requires real-runtime llama_cpp "
                "evidence; offline readiness does not run model evals."
                if production_real_runtime_eval_required
                else (
                    "Overlay activation requires baseline-versus-candidate behavioral A/B "
                    "evaluation. Injected deterministic clients may test the machinery, but "
                    "their evidence is not production evidence."
                )
            ),
            action=_VERIFY_REAL if production_real_runtime_eval_required else None,
        )
    )
    trainer = settings.finetune.trainer_executable
    evaluator = settings.finetune.evaluator_executable
    trainer_ready = bool(trainer and settings.resolve_path(trainer).is_file())
    evaluator_ready = bool(evaluator and settings.resolve_path(evaluator).is_file())
    if not settings.finetune.enabled:
        fine_tuning_status = "disabled"
        fine_tuning_check_status: CheckStatus = "skipped"
        fine_tuning_detail = "Fine-tuning is intentionally disabled."
    elif trainer_ready and evaluator_ready:
        fine_tuning_status = "ready"
        fine_tuning_check_status = "ok"
        fine_tuning_detail = "Reviewed trainer and evaluator executables are configured."
    else:
        fine_tuning_status = "unconfigured"
        fine_tuning_check_status = "warning"
        fine_tuning_detail = (
            "Fine-tuning is enabled but trainer/evaluator configuration is incomplete."
        )
    checks.append(
        ReadinessCheck(
            name="fine-tuning readiness",
            status=fine_tuning_check_status,
            detail=fine_tuning_detail,
            action=None if fine_tuning_status == "ready" else "run april finetune doctor",
        )
    )
    checks.append(
        ReadinessCheck(
            name="production app and Apple verification",
            status="skipped",
            detail=(
                "build=not_evaluated; signing=not_evaluated; "
                "notarization=not_evaluated; stapling=not_evaluated; "
                "gatekeeper=not_evaluated. Offline readiness did not run Apple tools."
            ),
            action="run april package build --output dist/APRIL.app --version VERSION",
        )
    )
    daemon_status_payload = _daemon_status(settings)
    daemon_status = str(daemon_status_payload.get("status", "unknown"))
    daemon_details_available = bool(daemon_status_payload.get("details_available", False))
    checks.append(
        ReadinessCheck(
            name="daemon detailed status",
            status="ok" if daemon_details_available else "warning",
            detail=(
                f"details available; daemon status={daemon_status}"
                if daemon_details_available
                else f"details unavailable; daemon status={daemon_status}"
            ),
            action="run april daemon status" if not daemon_details_available else None,
        )
    )

    # --- aggregate -----------------------------------------------------------
    blockers = [check.name for check in checks if check.status == "blocker"]
    warnings = [check.name for check in checks if check.status == "warning"]
    # Voice readiness is its own axis; model blockers are the voice "voice:" rows.
    model_blockers = [name for name in blockers if not name.startswith("voice:")]
    voice_blockers = [name for name in blockers if name.startswith("voice:")]
    real_model_preflight_ready = not model_blockers
    voice_preflight_ready = voice_enabled and not voice_blockers

    checks.append(
        ReadinessCheck(
            name="real-model verification",
            status="skipped",
            detail="Offline readiness did not load/chat/stream/unload a GGUF model.",
            action=_VERIFY_REAL,
        )
    )
    checks.append(
        ReadinessCheck(
            name="live voice verification",
            status="skipped",
            detail=(
                "Offline readiness did not run microphone/STT/TTS playback verification."
                if voice_enabled
                else "Voice disabled; live verification not requested."
            ),
            action=_VERIFY_VOICE if voice_enabled else None,
        )
    )

    next_actions: list[str] = []
    for check in checks:
        if check.action and check.action not in next_actions:
            next_actions.append(check.action)
    # Always end with the authoritative real verification command.
    if _VERIFY_REAL not in next_actions:
        next_actions.append(_VERIFY_REAL)
    if voice_enabled and _VERIFY_VOICE not in next_actions:
        next_actions.append(_VERIFY_VOICE)
    if voice_enabled and _VERIFY_WAKE not in next_actions:
        next_actions.append(_VERIFY_WAKE)
    if voice_enabled and _VERIFY_VOICE_CONVERSATION not in next_actions:
        next_actions.append(_VERIFY_VOICE_CONVERSATION)

    return ReadinessReport(
        generated_at=utc_now_iso(),
        os=f"{platform.system()} {platform.release()}".strip(),
        cpu_architecture=platform.machine(),
        python_version=platform.python_version(),
        runtime_backend=backend,
        runtime_is_fake=runtime_is_fake,
        llama_cpp_python_available=llama_available,
        environment=settings.environment,
        voice_enabled=voice_enabled,
        real_model_preflight_ready=real_model_preflight_ready,
        voice_preflight_ready=voice_preflight_ready,
        models=models,
        voice_artifacts=voice_artifacts,
        api_token_status=api_status,
        runtime_token_status=runtime_status,
        credential_store_selected=credential_store_selected,
        legacy_plaintext_credential_detected=legacy_plaintext,
        audit_chain_status=audit_status,
        database_quick_check=database_status.quick_check,
        database_foreign_key_consistent=database_status.foreign_key_consistent,
        database_wal_state=database_status.journal_mode,
        database_integrity_failures=list(database_status.failures),
        last_successful_backup=database_status.last_successful_backup,
        speaker_gate=settings.wake.speaker_gate,
        speaker_gate_supported=speaker_gate_supported,
        daemon_status=daemon_status,
        daemon_details_available=daemon_details_available,
        sentinel_live_status=sentinel_live_status,
        voice_conversation_live_status=voice_conversation_live_status,
        embedding_provider=settings.memory.embedding_provider,
        lexical_tokenizer_version="unicode-nfkc-casefold-v1",
        hashed_token_implementation_version="hashed-token-unicode-v2",
        hybrid_retrieval_enabled=True,
        runtime_batch_embedding_supported=True,
        runtime_batch_embedding_max_items=64,
        embedding_role_model_registered=bool(embedding_role_models),
        reasoning_role_model_registered=bool(reasoning_role_models),
        reasoning_falls_back_to_brain=not bool(reasoning_role_models),
        conversation_summarization_enabled=settings.conversation_context.summary_enabled,
        reading_model_registered=bool(reading_models),
        router_model_id=router_model_id,
        router_aliased_to_brain=router_aliased,
        dedicated_router_available=dedicated_router_available,
        router_failure_reason=router_failure_reason,
        conversation_summarization_available=reading_available,
        conversation_summarization_degrades_safely=True,
        hashed_token_embedding_fallback=settings.memory.embedding_provider == "hashed-token",
        lora_adapter_missing_count=lora_adapter_missing_count,
        adapter_lifecycle_consistent=bool(adapter_state["consistent"]),
        incomplete_adapter_operation_count=(
            adapter_state["incomplete_operation_count"]
            if isinstance(adapter_state["incomplete_operation_count"], int)
            else 0
        ),
        evolution_rollout_status=rollout_status,
        incomplete_rollout_transition_count=int(rollout_state["incomplete_transition_count"]),
        active_canary_count=int(rollout_state["active_canary_count"]),
        expired_canary_count=int(rollout_state["expired_canary_count"]),
        rollout_candidate_unavailable_count=int(rollout_state["candidate_unavailable_count"]),
        rollout_candidate_hash_mismatch_count=int(rollout_state["candidate_hash_mismatch_count"]),
        rollout_pointer_database_disagreement_count=int(
            rollout_state["pointer_database_disagreement_count"]
        ),
        rollout_rollback_required_count=int(rollout_state["rollback_required_count"]),
        lora_canary_supported=bool(rollout_state["lora_canary_supported"]),
        overlay_eval_mode=(
            "deterministic_fixture_plus_real_runtime"
            if production_real_runtime_eval_required
            else "deterministic_fixture"
        ),
        production_real_runtime_eval_required=production_real_runtime_eval_required,
        pending_real_runtime_overlay_blocker_count=len(pending_real_runtime_blockers),
        pending_real_runtime_overlay_blockers=pending_real_runtime_blockers,
        fine_tuning_status=fine_tuning_status,
        production_app_status="not_evaluated",
        signing_status="not_evaluated",
        notarization_status="not_evaluated",
        stapling_status="not_evaluated",
        gatekeeper_status="not_evaluated",
        apple_release_evidence_status="not_evaluated",
        evolution_enabled=settings.evolution.enabled,
        evolution_kill_switch_active=(settings.evolution_path / "DISABLED").exists(),
        scheduler_enabled=settings.scheduler.enabled,
        dreamer_last_report_available=any((settings.evolution_path / "reports").glob("*.json")),
        pending_eval_case_count=pending_evals,
        pending_write_capable_overlay_count=pending_overlays,
        model_import_uses_durable_jobs=True,
        memory_reindex_uses_durable_jobs=True,
        last_successful_semantic_reindex=(
            str(vector_metadata.get("last_successful_reindex_at"))
            if vector_metadata.get("provider") == "runtime-local"
            and isinstance(vector_metadata.get("last_successful_reindex_at"), str)
            else None
        ),
        active_vector_generation=(
            str(vector_metadata["active_generation"])
            if isinstance(vector_metadata.get("active_generation"), str)
            else None
        ),
        active_embedding_provider=(
            str(vector_metadata["provider"])
            if isinstance(vector_metadata.get("provider"), str)
            else settings.memory.embedding_provider
        ),
        active_embedding_model_id=(
            str(vector_metadata["embedding_model_id"])
            if isinstance(vector_metadata.get("embedding_model_id"), str)
            else settings.memory.embedding_model_id
        ),
        comparison_fixtures_installed=bool(fixture_metadata["installed"]),
        comparison_fixture_set_version=str(fixture_metadata["version"]),
        comparison_fixture_set_sha256=(
            str(fixture_metadata["sha256"]) if fixture_metadata["sha256"] else None
        ),
        real_benchmark_evidence_exists=bool(benchmark_evidence["exists"]),
        benchmark_evidence_current_hardware=bool(benchmark_evidence["current_hardware"]),
        benchmark_evidence_simulated=bool(benchmark_evidence["simulated"]),
        benchmark_evidence_stale=bool(benchmark_evidence["stale"]),
        benchmark_evidence_incomplete=bool(benchmark_evidence["incomplete"]),
        benchmark_evidence_production_eligible=bool(benchmark_evidence["production_eligible"]),
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        next_actions=next_actions,
    )
