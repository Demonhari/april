from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import platform
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from april_common.config_fingerprint import config_fingerprint_digest
from april_common.errors import AprilError
from april_common.process_environment import PROCESS_ENVIRONMENT_POLICY_VERSION
from april_common.process_runner import ResourceLimitProfile, resource_limit_report
from april_common.process_sandbox import SandboxBackend, sandbox_capabilities
from april_common.service_health import ServiceHealthResult, probe_service_health
from april_common.settings import (
    INSECURE_API_TOKENS,
    INSECURE_RUNTIME_TOKENS,
    AprilSettings,
)
from april_common.token_setup import legacy_plaintext_credentials_detected
from april_common.verification_evidence import verified_model_ids
from services.api.dependencies import ApiContainer
from services.api.model_readiness import model_registry_readiness as _model_registry_readiness
from services.api.production_activation import (
    finetuning_readiness,
    production_activation_failure_reasons,
)
from services.api.reporting import (
    _basename,
    _is_relative_to,
    _latest_live_voice_flags,
    _redact_path_text,
    _reports_freshness,
)
from services.april_runtime.model_registry import ModelRegistry
from services.evolution.adapters import AdapterLifecycleManager
from services.evolution.approval import PromptOverlayApprovalService
from services.evolution.dreamer import latest_report
from services.evolution.feedback_eval import count_pending_eval_cases
from services.evolution.inspect import (
    count_pending_write_capable_overlay_candidates,
    evolution_kill_switch_active,
)
from services.evolution.rollouts import RolloutService
from services.memory.maintenance import check_database
from services.memory.migrations import SCHEMA_VERSION
from services.tool_worker.limits import UnsafeToolWorkerSocket, validate_live_socket
from services.voice.health import microphone_access, query_audio_devices, voice_readiness_summary


def _redact_health_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"path", "model_path", "binary_path"}:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_health_payload(item)
        return redacted
    if isinstance(value, list):
        return [_redact_health_payload(item) for item in value]
    return value


async def readiness_payload(
    active: ApiContainer,
    *,
    probe_service_health_fn: Callable[..., ServiceHealthResult] = probe_service_health,
) -> dict[str, Any]:
    runtime_status = "unavailable"
    runtime_backend = "unknown"
    runtime_simulated: bool | None = None
    runtime_health: dict[str, Any]
    runtime_probe = await asyncio.to_thread(
        probe_service_health_fn,
        active.settings.runtime.url.rstrip("/") + "/runtime/health",
        bearer_token=active.settings.runtime.token,
        timeout=1.0,
    )
    if runtime_probe.ok:
        try:
            raw_runtime = await active.runtime_client.health(timeout=1.0)
            runtime_health = _safe_runtime_health(raw_runtime)
            runtime_status = str(raw_runtime.get("status", "unknown"))
            runtime_backend = str(raw_runtime.get("backend", "unknown"))
            simulated = raw_runtime.get("simulated")
            runtime_simulated = simulated if isinstance(simulated, bool) else None
        except AprilError as exc:
            runtime_health = {"status": "unavailable", "error": exc.message}
            runtime_probe = ServiceHealthResult(
                ok=False,
                status_code=None,
                reason="invalid_response",
                message="Runtime health response could not be read.",
            )
    else:
        runtime_health = {
            "status": "unavailable",
            "probe_reason": runtime_probe.reason,
            "http_status": runtime_probe.status_code,
            "error": runtime_probe.message,
        }

    if runtime_probe.ok:
        try:
            raw_models = await active.runtime_client.models()
            models = [
                _safe_model_entry(model, runtime_backend) for model in raw_models.get("models", [])
            ]
        except AprilError:
            models = []
    else:
        models = []
    if not models and isinstance(runtime_health.get("models"), list):
        models = [
            _safe_model_entry(model, runtime_backend)
            for model in runtime_health.get("models", [])
            if isinstance(model, dict)
        ]

    vector_health = active.vector_memory.health()
    memory_index_health = await active.memory_repository.health()
    vector = _redact_health_payload(vector_health)
    configured_embedding_provider = active.settings.memory.embedding_provider
    active_embedding_provider = str(vector_health.get("embedding", "hashed-token"))
    embedding_index_compatible = bool(vector_health.get("compatible", True))
    embedding_model_status = _embedding_model_status(active.settings)
    fell_back_to_hashed_token = (
        configured_embedding_provider == "runtime-local"
        and active_embedding_provider == "hashed-token"
    )
    embedding_warnings: list[str] = []
    if (
        active.settings.environment == "production"
        and configured_embedding_provider != "runtime-local"
    ):
        embedding_warnings.append(
            "runtime-local embeddings are not configured in production-like mode"
        )
    if fell_back_to_hashed_token:
        embedding_warnings.append("runtime-local embeddings fell back to hashed-token")
    if not embedding_model_status["embedding_model_registered"]:
        embedding_warnings.append("no embedding-role model is registered")
    embeddings = {
        "configured_provider": configured_embedding_provider,
        "active_provider": active_embedding_provider,
        "runtime_local_requested": configured_embedding_provider == "runtime-local",
        "fell_back_to_hashed_token": fell_back_to_hashed_token,
        "hashed_token_active": active_embedding_provider == "hashed-token",
        "hashed_token_fallback_active": fell_back_to_hashed_token,
        "embedding_model_id": active.settings.memory.embedding_model_id,
        "dimensions": vector_health.get("dimensions"),
        "index_compatible": embedding_index_compatible,
        "persisted_provider": vector_health.get("persisted_provider"),
        "reindex_required": not embedding_index_compatible,
        "reindex_command": "run april memory reindex",
        "warnings": embedding_warnings,
        "active_generation": vector_health.get("effective_generation"),
        "last_successful_reindex_at": vector_health.get("last_successful_reindex_at"),
        "vector_index_status": vector_health.get("status"),
        "repair_command": vector_health.get("repair_command"),
    }
    embeddings.update(embedding_model_status)
    # query_audio_devices() only *enumerates* devices; it never opens the
    # microphone or starts a stream. Readiness stays inert by construction.
    devices = query_audio_devices()
    voice_readiness = voice_readiness_summary(active.settings, devices)
    # Lift the offline milestone to a live rung only when a redacted live report
    # proves it. wake_live_verified outranks live_verified.
    live_flags = _latest_live_voice_flags(active.settings)
    voice_milestone = str(voice_readiness.get("voice_milestone", "not_configured"))
    if active.settings.voice.enabled:
        if live_flags["voice_conversation_live_verified"]:
            voice_milestone = "conversation_live_verified"
        elif live_flags["wake_word_live_verified"]:
            voice_milestone = "wake_live_verified"
        elif live_flags["voice_live_verified"]:
            voice_milestone = "live_verified"
    voice_artifacts = [
        _voice_artifact(
            active.settings,
            "wake confirmation whisper binary",
            active.settings.voice.effective_confirmation_whisper_binary_path,
        ),
        _voice_artifact(
            active.settings,
            "wake confirmation whisper model",
            active.settings.voice.effective_confirmation_whisper_model_path,
        ),
        _voice_artifact(
            active.settings,
            "transcription whisper binary",
            active.settings.voice.effective_transcription_whisper_binary_path,
        ),
        _voice_artifact(
            active.settings,
            "transcription whisper model",
            active.settings.voice.effective_transcription_whisper_model_path,
        ),
        _voice_artifact(active.settings, "piper binary", active.settings.voice.piper_binary_path),
        _voice_artifact(active.settings, "piper model", active.settings.voice.piper_model_path),
    ]
    wake_word_model_paths = _wake_word_model_artifacts(active.settings)
    voice_artifacts.extend(wake_word_model_paths)
    api_localhost = active.settings.api.host in {"127.0.0.1", "localhost"}
    runtime_localhost = active.settings.runtime.url.startswith(
        ("http://127.0.0.1", "http://localhost")
    )
    try:
        database_available = (await active.database.fetchone("SELECT 1")) is not None
    except Exception:
        database_available = False
    database_integrity = await asyncio.to_thread(
        check_database,
        active.settings.database_path,
        home=active.settings.home,
    )
    try:
        audit_size = (
            active.settings.audit_path.stat().st_size if active.settings.audit_path.exists() else 0
        )
        audit_chain_status = (
            active.approvals.audit.verify().status
            if audit_size <= 4 * 1024 * 1024
            else "explicit_verification_required"
        )
    except (OSError, RuntimeError, AprilError):
        audit_chain_status = "unavailable"
    legacy_plaintext = legacy_plaintext_credentials_detected(active.settings.home)
    credential_store_selected: str = active.settings.security.credential_store
    if credential_store_selected == "auto":
        credential_store_selected = (
            "macos-keychain"
            if active.settings.environment == "production" and platform.system() == "Darwin"
            else "legacy-development-default"
        )
    model_registry = _model_registry_readiness(active.settings)
    summary_readiness = _conversation_summary_readiness(
        active,
        runtime_available=runtime_probe.ok,
    )
    adapter_state = await AdapterLifecycleManager(
        active.settings,
        active.database,
        audit=active.approvals.audit,
    ).state_health()
    job_counts = (
        await active.job_store.counts()
        if active.job_store is not None
        else {"queued": 0, "running": 0, "interrupted": 0, "expired_leases": 0}
    )
    tool_worker_live = bool(
        active.tool_worker_manager is not None
        and active.tool_worker_manager.process is not None
        and active.tool_worker_manager.process.returncode is None
    )
    tool_worker_socket_mode: str | None = None
    tool_worker_protocol_ready = False
    tool_worker_self_check = False
    if active.tool_worker_manager is not None:
        try:
            tool_worker_socket_mode = validate_live_socket(
                active.tool_worker_manager.socket_path,
                runtime_directory=active.tool_worker_manager.runtime_directory,
            )
            tool_worker_protocol_ready = active.tool_worker_client is not None
            if active.tool_worker_client is not None and active.settings.allowed_roots:
                response = await active.tool_worker_client.self_check(
                    project_root=active.settings.allowed_roots[0]
                )
                tool_worker_self_check = response.ok
        except (OSError, UnsafeToolWorkerSocket, Exception):
            tool_worker_protocol_ready = False
            tool_worker_self_check = False
    job_worker_live = bool(
        active.job_worker_manager is not None
        and active.job_worker_manager.process is not None
        and active.job_worker_manager.process.returncode is None
    )
    job_worker_ready = bool(
        active.job_worker_manager is not None and active.job_worker_manager.status_path.exists()
    )
    scheduler_required = active.settings.scheduler.enabled
    scheduler_available = active.scheduler is not None and active.scheduler.running
    failure_reasons = _readiness_failure_reasons(
        runtime_probe=runtime_probe,
        runtime_status=runtime_status,
        database_available=database_available,
        model_registry=model_registry,
        scheduler_required=scheduler_required,
        scheduler_available=scheduler_available,
        vector_health=vector_health,
    )
    sandbox = sandbox_capabilities(
        environment=active.settings.environment,
        development_override=(active.settings.workers.development_unsandboxed_override),
    )
    if (
        active.settings.environment == "production"
        and sandbox.backend is SandboxBackend.UNAVAILABLE
    ):
        failure_reasons.append(
            {
                "code": "process_sandbox_unavailable",
                "message": "Risky subprocess operations fail closed without an OS sandbox.",
            }
        )
    if not bool(adapter_state["consistent"]):
        failure_reasons.append(
            {
                "code": "adapter_state_inconsistent",
                "message": "Adapter lifecycle state requires reconciliation.",
            }
        )
    rollout_state = await RolloutService(
        active.settings,
        active.database,
        audit=active.approvals.audit,
    ).health()
    if rollout_state["status"] == "degraded":
        failure_reasons.append(
            {
                "code": "evolution_rollout_unsafe",
                "message": (
                    "An incomplete rollout, expired canary, artifact integrity "
                    "failure, or active-pointer disagreement requires rollback."
                ),
                "action": str(rollout_state.get("action") or "run april evolve rollout list"),
            }
        )
    if active.settings.workers.tool_worker_enabled and (
        not tool_worker_protocol_ready or not tool_worker_self_check
    ):
        failure_reasons.append(
            {
                "code": "tool_worker_unavailable",
                "message": "Tool Worker is not ready; risky tools fail closed.",
            }
        )
    if active.settings.workers.job_worker_enabled and not job_worker_ready:
        failure_reasons.append(
            {
                "code": "job_worker_unavailable",
                "message": "Job Worker is not ready; durable jobs remain queued.",
            }
        )
    ready = not failure_reasons
    overlay_approval_service = PromptOverlayApprovalService(
        active.settings,
        active.database,
        audit=active.approvals.audit,
        runtime_client=active.runtime_client,
    )
    pending_write_capable_overlays = await overlay_approval_service.list_pending()
    pending_real_runtime_blockers = _pending_real_runtime_overlay_blockers(active.settings)
    real_runtime_required = active.settings.environment == "production"
    finetuning = finetuning_readiness(active.settings)
    production_failure_reasons = production_activation_failure_reasons(
        settings=active.settings,
        runtime_probe=runtime_probe,
        runtime_backend=runtime_backend,
        runtime_simulated=runtime_simulated,
        model_registry=model_registry,
        verified_models=verified_model_ids(active.settings.home),
        embeddings=embeddings,
        live_flags=live_flags,
        job_worker_ready=job_worker_ready,
        tool_worker_protocol_ready=tool_worker_protocol_ready,
        tool_worker_self_check=tool_worker_self_check,
        audit_chain_status=audit_chain_status,
        database_integrity=database_integrity,
        rollout_state=rollout_state,
        finetuning=finetuning,
        credential_store_selected=credential_store_selected,
        legacy_plaintext_credential_detected=legacy_plaintext,
    )
    return {
        "ready": ready,
        "status": "ok" if ready else "degraded",
        "readiness_scope": "operational",
        "production_ready": not production_failure_reasons,
        "production_failure_reasons": production_failure_reasons,
        "failure_reasons": failure_reasons,
        "core": {
            "api_health": "ok",
            "runtime_health": runtime_status,
            "runtime_backend": runtime_backend,
            "runtime_simulated": runtime_simulated,
            "runtime": runtime_health,
            "database": {
                "status": "ok" if database_available else "unavailable",
                "configured": True,
                "available": database_available,
            },
            "vector_index": vector,
            "scheduler": {
                "enabled": active.settings.scheduler.enabled,
                "running": active.scheduler.running if active.scheduler else False,
                "briefing_enabled": active.settings.scheduler.briefing_enabled,
            },
            "required_services": {
                "runtime": runtime_probe.ok,
                "database": database_available,
                "scheduler": not scheduler_required or scheduler_available,
            },
        },
        "memory_index": asdict(memory_index_health),
        "models": {
            "llama_cpp_python_available": importlib.util.find_spec("llama_cpp") is not None,
            "registered": models,
            "lora_adapters": _lora_adapter_readiness(active.settings),
            "adapter_lifecycle": adapter_state,
            "registry": model_registry,
        },
        "embeddings": embeddings,
        "conversation_summarization": summary_readiness,
        "jobs": {
            "schema_version": SCHEMA_VERSION,
            "worker_enabled": active.settings.workers.job_worker_enabled,
            "queued": job_counts.get("queued", 0),
            "running": job_counts.get("running", 0),
            "interrupted": job_counts.get("interrupted", 0),
            "expired_leases": job_counts.get("expired_leases", 0),
            "worker_liveness": job_worker_live,
            "worker_readiness": job_worker_ready,
        },
        "tool_worker": {
            "enabled": active.settings.workers.tool_worker_enabled,
            "process_liveness": tool_worker_live,
            "socket_available": tool_worker_socket_mode is not None,
            "socket_mode": tool_worker_socket_mode,
            "protocol_readiness": tool_worker_protocol_ready,
            "self_check": tool_worker_self_check,
        },
        "process_policy": {
            "environment_policy_version": PROCESS_ENVIRONMENT_POLICY_VERSION,
            "sandbox_policy_version": sandbox.policy_version,
            "sandbox_backend": sandbox.backend.value,
            "network_denial_available": sandbox.network_denial_available,
            "filesystem_policy_available": sandbox.filesystem_policy_available,
            "production_fail_closed": sandbox.production_fail_closed,
            "development_unsandboxed_override": sandbox.development_override_enabled,
            "sandbox_warning": sandbox.warning,
            "unsupported_resource_limits": list(
                resource_limit_report(ResourceLimitProfile.COMMAND).unsupported
            ),
        },
        "evolution": {
            "enabled": active.settings.evolution.enabled,
            "kill_switch_active": evolution_kill_switch_active(active.settings),
            "scheduler_enabled": active.settings.scheduler.enabled,
            "scheduler_running": active.scheduler.running if active.scheduler else False,
            "overlay_eval_mode": (
                "deterministic_fixture_plus_real_runtime"
                if real_runtime_required
                else "deterministic_fixture"
            ),
            "deterministic_fixture_eval_kind": "deterministic_fixture",
            "real_runtime_eval_required": real_runtime_required,
            "pending_real_runtime_overlay_blocker_count": len(pending_real_runtime_blockers),
            "pending_real_runtime_overlay_blockers": pending_real_runtime_blockers,
            "pending_write_capable_overlay_approval_count": len(pending_write_capable_overlays),
            "pending_write_capable_overlay_candidate_count": (
                count_pending_write_capable_overlay_candidates(active.settings)
            ),
            "pending_eval_case_count": count_pending_eval_cases(active.settings),
            "rollouts": rollout_state,
        },
        "fine_tuning": finetuning,
        "release_evidence": {
            "production_app_build": "not_evaluated",
            "signing": "not_evaluated",
            "notarization": "not_evaluated",
            "stapling": "not_evaluated",
            "gatekeeper": "not_evaluated",
            "production_ready": False,
        },
        # Redacted local config digest + per-type report freshness, so the Desktop
        # operator console and `doctor --daily-driver` can flag stale reports.
        "config_fingerprint": config_fingerprint_digest(active.settings.home),
        "reports": _reports_freshness(active.settings),
        "verification_guidance": {
            "commands": [
                "run april verify --all-configured-models --require-real-model "
                "--report data/verification/mac-readiness.json",
                "run april verify --workflow --real-model "
                "--report data/verification/workflow-real.json",
                "run april verify /absolute/path/to/model.gguf --target-mac "
                "--require-real-model --report data/verification/single-model.json",
                "run april voice verify-wake-live --report data/verification/wake-live.json",
            ],
            "warnings": [
                "Fake verification is not real model verification.",
                "Desktop never loads models or starts voice automatically.",
                "Reports are redacted and show model basenames only.",
                "Generated verification reports and app stubs are ignored by Git.",
            ],
        },
        "voice": {
            "enabled": active.settings.voice.enabled,
            "sounddevice_available": bool(devices.get("sounddevice_installed")),
            "microphone_access": microphone_access(devices)["status"],
            "input_device_count": len(devices.get("input_devices", [])),
            "output_device_count": len(devices.get("output_devices", [])),
            "macos_microphone_permission_guidance": (
                "macOS: System Settings > Privacy & Security > Microphone. "
                "Allow the terminal app used to run APRIL."
            ),
            "artifacts": voice_artifacts,
            "wake_word_model_paths": wake_word_model_paths,
            "wake_live_report_status": (
                "verified" if live_flags["wake_word_live_verified"] else "not_verified"
            ),
            "wake_live_report_missing": not live_flags["wake_word_live_verified"],
            "push_to_talk_available_without_wake_word": True,
            "openwakeword_available": voice_readiness["openwakeword_available"],
            "push_to_talk_ready": voice_readiness["push_to_talk_ready"],
            "wake_word_ready": voice_readiness["wake_word_ready"],
            "full_voice_loop_ready": voice_readiness["full_voice_loop_ready"],
            "sentinel_live_verified": live_flags["wake_word_live_verified"],
            "conversation_endpointing_configured": True,
            "endpoint_silence_ms": active.settings.voice.endpoint_silence_ms,
            "minimum_utterance_ms": active.settings.voice.minimum_utterance_ms,
            "barge_in_trigger": active.settings.voice.barge_in_trigger,
            "barge_in_action": active.settings.voice.barge_in_action,
            "acoustic_echo_cancellation_available": False,
            "complete_live_conversation_verified": live_flags["voice_conversation_live_verified"],
            "complete_live_conversation_command": (
                "run april voice verify-conversation-live "
                "--report data/verification/voice-conversation-live.json"
            ),
            "speaker_gate": {
                "mode": active.settings.wake.speaker_gate,
                "supported": False,
                "detail": (
                    (
                        "speaker_gate=soft is configured, but no production local "
                        "speaker verifier model ships with APRIL; Sentinel audits one "
                        "warning and behaves as off."
                        if active.settings.wake.speaker_gate == "soft"
                        else "speaker_gate is off; enrollment does not enable it by itself."
                    )
                    + " It is a convenience filter, never a security boundary."
                ),
            },
            # Single redacted enum capturing the highest voice milestone reached:
            # disabled / not_configured / push_to_talk_ready / wake_word_ready /
            # full_voice_loop_ready / live_verified / wake_live_verified.
            "voice_milestone": voice_milestone,
        },
        "security": {
            "allowed_filesystem_roots": [
                {
                    "basename": root.name or str(root),
                    "exists": root.exists(),
                    "within_april_home": _is_relative_to(root, active.settings.home),
                }
                for root in active.settings.allowed_roots
            ],
            "api_token": {"status": "configured" if active.settings.api.token else "missing"},
            "runtime_token": {
                "status": "configured" if active.settings.runtime.token else "missing"
            },
            "credential_store": credential_store_selected,
            "legacy_plaintext_credential_detected": legacy_plaintext,
            "audit_chain_status": audit_chain_status,
            "database_integrity": {
                "ok": database_integrity.ok,
                "quick_check": database_integrity.quick_check,
                "foreign_key_consistent": database_integrity.foreign_key_consistent,
                "wal_state": database_integrity.journal_mode,
                "schema_version": database_integrity.schema_version,
                "expected_schema_version": database_integrity.expected_schema_version,
                "migration_consistent": database_integrity.migration_consistent,
                "failures": list(database_integrity.failures),
                "last_successful_backup": database_integrity.last_successful_backup,
                "checked_at": database_integrity.checked_at,
            },
            "api_localhost_binding": api_localhost,
            "runtime_localhost_binding": runtime_localhost,
            "cors_enabled": active.settings.api.cors_enabled,
            "development_token_warning": _development_token_warning(active.settings),
        },
        "daemon": _daemon_readiness(active.settings),
        "next_actions": [
            "run april verify --all-configured-models --require-real-model "
            "--report data/verification/mac-readiness.json",
            "run april voice verify-live --report data/verification/voice-live.json",
            "run april voice verify-wake-live --report data/verification/wake-live.json",
            "run april finetune doctor",
            "run april package build --output dist/APRIL.app --version VERSION",
            'run april package sign dist/APRIL.app --identity "Developer ID Application: NAME"',
            "run april package notarize-submit dist/APRIL.zip --keychain-profile APRIL_NOTARY",
        ],
    }


def _conversation_summary_readiness(
    active: ApiContainer, *, runtime_available: bool
) -> dict[str, Any]:
    enabled = active.settings.conversation_context.summary_enabled
    reading_agent = active.agent_registry.get("reading_agent")
    model_id = reading_agent.model_id if reading_agent is not None else None
    model_entry_exists = False
    model_artifact_available = False
    if model_id:
        try:
            registry = ModelRegistry.from_file(
                active.settings.home / "configs" / "models.yaml",
                root=active.settings.home,
            )
            model_entry_exists = registry.exists(model_id)
            if model_entry_exists:
                model = registry.get(model_id)
                model_artifact_available = (
                    active.settings.runtime.backend == "fake"
                    or model.backend == "fake"
                    or model.resolved_path(registry.root).is_file()
                )
        except AprilError:
            pass
    available = bool(
        enabled
        and model_id
        and model_entry_exists
        and model_artifact_available
        and runtime_available
    )
    return {
        "enabled": enabled,
        "reading_agent_configured": reading_agent is not None,
        "model_entry_exists": model_entry_exists,
        "available": available,
        "degrades_safely": True,
        "status": ("disabled" if not enabled else ("available" if available else "degraded")),
    }


def _readiness_failure_reasons(
    *,
    runtime_probe: ServiceHealthResult,
    runtime_status: str,
    database_available: bool,
    model_registry: dict[str, Any],
    scheduler_required: bool,
    scheduler_available: bool,
    vector_health: dict[str, Any],
) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    if not runtime_probe.ok:
        if runtime_probe.reason == "authentication_rejected":
            message = "Runtime authentication was rejected."
        elif runtime_probe.reason == "endpoint_not_found":
            message = "The configured Runtime health endpoint is missing or incorrect."
        else:
            message = "Runtime is not reachable."
        reasons.append({"code": f"runtime_{runtime_probe.reason}", "message": message})
    elif runtime_status not in {"ok", "degraded"}:
        reasons.append(
            {
                "code": "runtime_unhealthy",
                "message": "Runtime reported an unhealthy operational status.",
            }
        )
    if not database_available:
        reasons.append({"code": "database_unavailable", "message": "Database is unavailable."})
    if not bool(model_registry["valid"]):
        reasons.append({"code": "model_registry_invalid", "message": "Model registry is invalid."})
    elif not bool(model_registry["required_model_available"]):
        reasons.append(
            {
                "code": "required_model_unavailable",
                "message": "A required model is unavailable.",
            }
        )
    if scheduler_required and not scheduler_available:
        reasons.append(
            {
                "code": "required_scheduler_unavailable",
                "message": "The required scheduler service is unavailable.",
            }
        )
    vector_status = str(vector_health.get("status", "not_ready"))
    if vector_status == "not_ready":
        reasons.append(
            {
                "code": "vector_index_unavailable",
                "message": "Vector retrieval has no valid index generation.",
            }
        )
    elif bool(vector_health.get("fallback_active")):
        reasons.append(
            {
                "code": "vector_index_recovery_active",
                "message": "Vector retrieval is using a read-only recovery generation.",
            }
        )
    elif vector_status == "degraded":
        reasons.append(
            {
                "code": "vector_index_degraded",
                "message": "Vector retrieval requires index repair or reindexing.",
            }
        )
    return reasons


def _embedding_model_status(settings: AprilSettings) -> dict[str, Any]:
    model_id = settings.memory.embedding_model_id
    status: dict[str, Any] = {
        "embedding_model_registered": False,
        "embedding_model_path_exists": False,
        "embedding_model_missing_reason": None,
    }
    try:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml",
            root=settings.home,
        )
    except Exception:
        status["embedding_model_missing_reason"] = "model registry is unavailable"
        return status
    candidates = [model for model in registry.list() if model.role == "embedding"]
    model = None
    if model_id:
        with contextlib.suppress(Exception):
            candidate = registry.get(model_id)
            if candidate.role == "embedding":
                model = candidate
    elif candidates:
        model = candidates[0]
    if model is None:
        if settings.memory.embedding_provider == "runtime-local":
            status["embedding_model_missing_reason"] = (
                "runtime-local requested without a registered role=embedding model"
                if not model_id
                else "embedding model id is not registered with role=embedding"
            )
        else:
            status["embedding_model_missing_reason"] = "no role=embedding model is registered"
        return status
    path = model.resolved_path(registry.root)
    status["embedding_model_registered"] = True
    status["embedding_model_path_exists"] = path.exists()
    if not path.exists():
        status["embedding_model_missing_reason"] = f"missing model file: {path.name}"
    return status


def _wake_word_model_artifacts(settings: AprilSettings) -> list[dict[str, Any]]:
    paths = settings.voice.effective_wake_word_model_paths
    if not paths:
        return [_voice_artifact(settings, "wake-word model", None)]
    artifacts: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        name = "wake-word model" if index == 0 else f"wake-word model {index + 1}"
        artifacts.append(_voice_artifact(settings, name, path))
    return artifacts


def _lora_adapter_readiness(settings: AprilSettings) -> list[dict[str, Any]]:
    try:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml",
            root=settings.home,
        )
    except Exception:
        return []
    adapters: list[dict[str, Any]] = []
    for model in registry.list():
        adapter = model.resolved_adapter_path(registry.root)
        if adapter is None:
            continue
        exists = adapter.exists()
        adapters.append(
            {
                "model_id": model.id,
                "configured": True,
                "missing": not exists,
                "basename": adapter.name,
                "status": "present_unverified" if exists else "missing_blocker",
                "detail": (
                    "adapter present; real-model verification still required"
                    if exists
                    else "configured adapter file is missing; model load fails closed"
                ),
            }
        )
    return adapters


def _pending_real_runtime_overlay_blockers(settings: AprilSettings) -> list[dict[str, str]]:
    report = latest_report(settings)
    if report is None:
        return []
    phases = report.get("phases")
    examine = phases.get("examine") if isinstance(phases, dict) else None
    pending = examine.get("pending_real_runtime") if isinstance(examine, dict) else None
    if not isinstance(pending, list):
        return []
    blockers: list[dict[str, str]] = []
    for item in pending:
        if not isinstance(item, dict):
            continue
        blockers.append(
            {
                "agent": str(item.get("agent") or "unknown"),
                "status": str(item.get("status") or "unknown"),
                "reason": _redact_path_text(
                    str(item.get("reason") or "real-runtime evaluation did not pass")
                )[:240],
            }
        )
    return blockers


def _daemon_readiness(settings: AprilSettings) -> dict[str, Any]:
    try:
        from apps.daemon.apriald import read_daemon_status

        payload = read_daemon_status(settings)
    except Exception:
        payload = {"status": "unknown", "details_available": False}
    return {
        "status": payload.get("status", "unknown"),
        "details_available": bool(payload.get("details_available", False)),
        "children": payload.get("children", []),
        "governor": payload.get("governor", {}),
    }


def _safe_runtime_health(payload: dict[str, Any]) -> dict[str, Any]:
    safe = _redact_health_payload(payload)
    if isinstance(safe, dict) and isinstance(safe.get("models"), list):
        backend = str(safe.get("backend", "unknown"))
        safe["models"] = [
            _safe_model_entry(model, backend) for model in safe["models"] if isinstance(model, dict)
        ]
    return safe if isinstance(safe, dict) else {"status": "unknown"}


def _safe_model_entry(model: dict[str, Any], runtime_backend: str) -> dict[str, Any]:
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


def _voice_artifact(settings: AprilSettings, name: str, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"name": name, "configured": False, "missing": True, "basename": None}
    resolved = settings.resolve_path(path)
    return {
        "name": name,
        "configured": True,
        "missing": not resolved.exists(),
        "basename": resolved.name,
    }


def _development_token_warning(settings: AprilSettings) -> str | None:
    if not settings.api.token or settings.api.token in INSECURE_API_TOKENS:
        return "API token uses an insecure development/placeholder default or is empty."
    if not settings.runtime.token or settings.runtime.token in INSECURE_RUNTIME_TOKENS:
        return "Runtime token uses an insecure development/placeholder default or is missing."
    return None
