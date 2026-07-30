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
from april_common.model_artifacts import gguf_artifact_status
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


def _build_model_and_registry_checks(
    root: Path,
    settings: AprilSettings,
    checks: list[ReadinessCheck],
) -> tuple[Any, ...]:
    backend = settings.runtime.backend
    runtime_is_fake = backend != "llama_cpp"
    llama_available = importlib.util.find_spec("llama_cpp") is not None

    # --- runtime backend -----------------------------------------------------
    if runtime_is_fake:
        checks.append(
            ReadinessCheck(
                name="runtime backend",
                status="blocker",
                detail=f"Backend is '{backend}' (fake/simulated), not 'llama_cpp'.",
                action="Set APRIL_RUNTIME_BACKEND=llama_cpp (or runtime.backend in april.yaml).",
            )
        )
    else:
        checks.append(ReadinessCheck(name="runtime backend", status="ok", detail="llama_cpp"))

    # --- llama-cpp-python extra ---------------------------------------------
    if llama_available:
        checks.append(
            ReadinessCheck(name="llama-cpp-python", status="ok", detail="import spec found")
        )
    else:
        checks.append(
            ReadinessCheck(
                name="llama-cpp-python",
                status="blocker",
                detail="Optional runtime extra is not installed.",
                action=_INSTALL_RUNTIME,
            )
        )

    # --- configured GGUF model files ----------------------------------------
    models: list[ReadinessModel] = []
    invalid_models: dict[str, str] = {}
    try:
        registry = ModelRegistry.from_file(root / "configs" / "models.yaml", root=root)
    except ConfigError as exc:
        checks.append(
            ReadinessCheck(
                name="model registry",
                status="blocker",
                detail=redact_reason(str(exc)),
                action="run april config validate",
            )
        )
        registry = None

    if registry is not None:
        for model in registry.list():
            path = model.resolved_path(registry.root)
            exists = path.exists()
            artifact_status = (
                gguf_artifact_status(path) if model.backend == "llama_cpp" else "not_applicable"
            )
            models.append(
                ReadinessModel(
                    id=model.id,
                    role=model.role,
                    backend=model.backend,
                    path_basename=path.name,
                    path_exists=exists,
                    artifact_status=artifact_status,
                )
            )
            if model.backend == "llama_cpp" and artifact_status != "valid":
                invalid_models[model.id] = artifact_status
        if invalid_models:
            checks.append(
                ReadinessCheck(
                    name="configured GGUF model files",
                    status="blocker",
                    detail="Unavailable or invalid model artifacts: "
                    + ", ".join(
                        f"{model_id} ({status})"
                        for model_id, status in sorted(invalid_models.items())
                    ),
                    action=_SETUP_MODELS,
                )
            )
        elif any(model.backend == "llama_cpp" for model in registry.list()):
            checks.append(
                ReadinessCheck(
                    name="configured GGUF model files",
                    status="ok",
                    detail="All required configured llama_cpp model files are present.",
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    name="configured GGUF model files",
                    status="warning",
                    detail="No llama_cpp model is configured; only fake models exist.",
                    action=_SETUP_MODELS,
                )
            )

    router_model_id = settings.brain.router_model_id or settings.brain.model_id
    router_aliased = settings.brain.router_model_id is None
    dedicated_router_available = False
    router_failure_reason: str | None = None
    if registry is not None:
        if router_aliased:
            if registry.exists(settings.brain.model_id):
                checks.append(
                    ReadinessCheck(
                        name="router model",
                        status="ok",
                        detail=f"{router_model_id} aliases the Brain model.",
                    )
                )
            else:
                router_failure_reason = "aliased_brain_model_not_registered"
        elif not registry.exists(router_model_id):
            router_failure_reason = "dedicated_router_not_registered"
        else:
            router_model = registry.get(router_model_id)
            if router_model.role != "router":
                router_failure_reason = "dedicated_router_role_mismatch"
            else:
                dedicated_router_available = (
                    backend == "fake"
                    or router_model.backend == "fake"
                    or router_model.resolved_path(registry.root).is_file()
                )
                if not dedicated_router_available:
                    router_failure_reason = "dedicated_router_artifact_unavailable"
        if router_failure_reason is not None:
            checks.append(
                ReadinessCheck(
                    name="router model",
                    status="blocker",
                    detail=router_failure_reason,
                    action="run april config validate",
                )
            )

    reading_models = (
        [model for model in registry.list() if model.role == "reading"]
        if registry is not None
        else []
    )
    reading_available = bool(
        reading_models
        and registry is not None
        and (
            backend == "fake"
            or any(
                model.backend == "fake" or model.resolved_path(registry.root).is_file()
                for model in reading_models
            )
        )
    )
    if not settings.conversation_context.summary_enabled:
        summary_check = ReadinessCheck(
            name="conversation summarization",
            status="skipped",
            detail=(
                "Conversation summarization is disabled by configuration; chat remains available."
            ),
        )
    elif reading_available:
        summary_check = ReadinessCheck(
            name="conversation summarization",
            status="ok",
            detail="A configured local reading-role model is available.",
        )
    else:
        summary_check = ReadinessCheck(
            name="conversation summarization",
            status="warning",
            detail=(
                "The optional local reading-role model is unavailable. Summary checkpoints "
                "do not advance, and chat degrades safely to the previous summary plus "
                "recent turns."
            ),
            action=_SETUP_MODELS,
        )
    checks.append(summary_check)

    # --- optional LoRA adapters (M15) ----------------------------------------
    if registry is not None:
        lora_adapter_missing_count = 0
        for model in registry.list():
            adapter = model.resolved_adapter_path(registry.root)
            if adapter is None:
                continue
            if adapter.exists():
                checks.append(
                    ReadinessCheck(
                        name=f"LoRA adapter: {model.id}",
                        status="warning",
                        detail=(
                            f"Adapter file {adapter.name} is present. LoRA serving is "
                            "wired but unverified until a real adapter is trained and "
                            "gated by a real-model verification report."
                        ),
                        action=_VERIFY_REAL,
                    )
                )
            else:
                lora_adapter_missing_count += 1
                checks.append(
                    ReadinessCheck(
                        name=f"LoRA adapter: {model.id}",
                        status="blocker",
                        detail=(
                            f"Missing adapter file: {adapter.name}. Model load fails "
                            "hard rather than silently serving the base model."
                        ),
                        action="Train or copy the adapter (see scripts/finetune/README.md).",
                    )
                )
    else:
        lora_adapter_missing_count = 0
    adapter_state = inspect_adapter_state(settings)
    if bool(adapter_state["consistent"]):
        checks.append(
            ReadinessCheck(
                name="adapter lifecycle state",
                status="ok",
                detail="Adapter pointer and database history are consistent.",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                name="adapter lifecycle state",
                status="blocker",
                detail=(
                    "An interrupted adapter operation or pointer/database "
                    "disagreement requires Core startup reconciliation."
                ),
                action="Restart the APRIL Core API, then run april readiness.",
            )
        )

    rollout_state = inspect_rollout_state(settings)
    rollout_status = str(rollout_state["status"])
    if rollout_status == "disabled":
        checks.append(
            ReadinessCheck(
                name="evolution rollout safety",
                status="skipped",
                detail="Evolution rollouts and canary traffic are intentionally disabled.",
            )
        )
    elif rollout_status in {"ok", "not_initialized"}:
        checks.append(
            ReadinessCheck(
                name="evolution rollout safety",
                status="ok",
                detail=(
                    "No unsafe rollout transition is present. LoRA canary remains "
                    "explicitly unsupported by the current Runtime lifecycle."
                ),
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                name="evolution rollout safety",
                status="blocker",
                detail=(
                    "An incomplete/expired rollout, artifact integrity failure, "
                    "or active-pointer disagreement requires rollback."
                ),
                action=str(rollout_state.get("action") or "run april evolve rollout list"),
            )
        )

    embedding_role_models = (
        [model for model in registry.list() if model.role == "embedding"]
        if registry is not None
        else []
    )
    reasoning_role_models = (
        [model for model in registry.list() if model.role == "reasoning"]
        if registry is not None
        else []
    )
    verified_model_ids = _verified_model_ids(root)
    if not reasoning_role_models:
        checks.append(
            ReadinessCheck(
                name="reasoning role readiness",
                status="warning",
                detail="Reasoning role is supported, but no local artifact is registered.",
                action=_IMPORT_REASONING,
            )
        )
        checks.append(
            ReadinessCheck(
                name="Deep and Council reasoning resolution",
                status="warning",
                detail=(
                    "No reasoning-role model is registered; Deep and Council reasoning "
                    "resolve to the Brain model."
                ),
                action=_IMPORT_REASONING,
            )
        )
    else:
        unverified_reasoning = [
            model.id for model in reasoning_role_models if model.id not in verified_model_ids
        ]
        checks.append(
            ReadinessCheck(
                name="reasoning role readiness",
                status="warning" if unverified_reasoning else "ok",
                detail=(
                    "Registered reasoning artifact has not passed real-model verification."
                    if unverified_reasoning
                    else "Registered reasoning artifact is present in a real-model report."
                ),
                action=(
                    "run april verify --all-configured-models --require-real-model"
                    if unverified_reasoning
                    else None
                ),
            )
        )
        checks.append(
            ReadinessCheck(
                name="Deep and Council reasoning resolution",
                status="ok" if not unverified_reasoning else "warning",
                detail=(
                    "Deep and Council resolve to a verified reasoning-role model."
                    if not unverified_reasoning
                    else "Deep and Council have a configured reasoning role, but its "
                    "real-model evidence is missing."
                ),
                action=_VERIFY_REAL if unverified_reasoning else None,
            )
        )

    checks.append(
        ReadinessCheck(
            name="embedding role registration",
            status="ok" if embedding_role_models else "warning",
            detail=(
                "At least one role=embedding model is registered."
                if embedding_role_models
                else "No role=embedding model is registered; hashed-token remains available."
            ),
            action=None if embedding_role_models else _IMPORT_EMBEDDING,
        )
    )

    # --- runtime-local embeddings ------------------------------------------
    if settings.memory.embedding_provider == "runtime-local":
        embedding_model_id = settings.memory.embedding_model_id
        if not embedding_model_id:
            checks.append(
                ReadinessCheck(
                    name="runtime-local embedding model",
                    status="blocker",
                    detail=(
                        "runtime-local embeddings are configured but no embedding model id is set."
                    ),
                    action=_SETUP_EMBEDDINGS,
                )
            )
        elif registry is None or not registry.exists(embedding_model_id):
            checks.append(
                ReadinessCheck(
                    name="runtime-local embedding model",
                    status="blocker",
                    detail=f"Embedding model id is not registered: {embedding_model_id}",
                    action=_SETUP_EMBEDDINGS,
                )
            )
        else:
            embedding_model = registry.get(embedding_model_id)
            if embedding_model.role != "embedding":
                checks.append(
                    ReadinessCheck(
                        name="runtime-local embedding model",
                        status="blocker",
                        detail=(
                            f"Configured embedding model id {embedding_model_id} has "
                            f"role={embedding_model.role}, not role=embedding."
                        ),
                        action=_SETUP_EMBEDDINGS,
                    )
                )
            else:
                embedding_path = embedding_model.resolved_path(registry.root)
                embedding_artifact_status = gguf_artifact_status(embedding_path)
                checks.append(
                    ReadinessCheck(
                        name="runtime-local embedding model",
                        status="ok" if embedding_artifact_status == "valid" else "blocker",
                        detail=(
                            f"Registered embedding model {embedding_model_id} exists."
                            if embedding_artifact_status == "valid"
                            else (
                                f"Embedding model {embedding_model_id} is "
                                f"{embedding_artifact_status}."
                            )
                        ),
                        action=(
                            None if embedding_artifact_status == "valid" else _SETUP_EMBEDDINGS
                        ),
                    )
                )
    else:
        checks.append(
            ReadinessCheck(
                name="runtime-local embedding model",
                status="skipped",
                detail="memory.embedding_provider is hashed-token; no embedding GGUF required.",
            )
        )

    active_vector_metadata = _active_vector_metadata(settings.vector_index_path)
    active_vector_provider = active_vector_metadata.get("provider")
    active_vector_model_id = active_vector_metadata.get("embedding_model_id")
    if settings.memory.embedding_provider == "hashed-token":
        checks.append(
            ReadinessCheck(
                name="semantic embedding generation",
                status="warning",
                detail="Hashed-token embeddings are active; semantic embedding is not enabled.",
                action=_IMPORT_EMBEDDING,
            )
        )
    elif embedding_role_models and (
        active_vector_provider != "runtime-local"
        or active_vector_model_id != settings.memory.embedding_model_id
        or not isinstance(active_vector_metadata.get("last_successful_reindex_at"), str)
    ):
        checks.append(
            ReadinessCheck(
                name="semantic embedding generation",
                status="warning",
                detail=(
                    "The active vector generation lacks a successful runtime-local reindex "
                    "for the configured embedding identity."
                ),
                action="run april memory reindex --wait",
            )
        )
    elif embedding_role_models:
        embedding_verified = all(model.id in verified_model_ids for model in embedding_role_models)
        checks.append(
            ReadinessCheck(
                name="semantic embedding generation",
                status="ok" if embedding_verified else "warning",
                detail=(
                    "Semantic embedding model and active vector generation are verified."
                    if embedding_verified
                    else "Semantic vector generation is active; model verification is pending."
                ),
                action=(
                    None
                    if embedding_verified
                    else "run april verify --all-configured-models --require-real-model"
                ),
            )
        )

    vector_metadata = active_vector_metadata
    fixture_metadata = fixture_set_metadata(settings.home)
    benchmark_evidence = _benchmark_evidence(settings)
    checks.extend(
        [
            ReadinessCheck(
                name="durable model import",
                status="ok",
                detail="`run april model import` requires exact approval and submits model_import.",
            ),
            ReadinessCheck(
                name="durable memory reindex",
                status="ok",
                detail="`run april memory reindex` submits memory_reindex.",
            ),
            ReadinessCheck(
                name="model comparison fixtures",
                status="ok" if fixture_metadata["installed"] else "warning",
                detail=(
                    f"Installed fixture set {fixture_metadata['version']}."
                    if fixture_metadata["installed"]
                    else "Versioned offline comparison fixtures are missing."
                ),
                action=(
                    None
                    if fixture_metadata["installed"]
                    else "Restore data/evaluations/model_benchmark/v1 from the APRIL source tree."
                ),
            ),
            ReadinessCheck(
                name="real model comparison evidence",
                status=("ok" if benchmark_evidence["production_eligible"] else "warning"),
                detail=(
                    "Production-eligible real evidence exists for this hardware profile."
                    if benchmark_evidence["production_eligible"]
                    else (
                        "Only fake/simulated comparison evidence exists."
                        if benchmark_evidence["simulated"]
                        else "Optional real comparison evidence is absent, stale, or incomplete."
                    )
                ),
                action=(
                    None
                    if benchmark_evidence["production_eligible"]
                    else (
                        "run april model compare-setups "
                        "--shared-model-id LOCAL_SHARED_MODEL_ID --wait"
                    )
                ),
            ),
        ]
    )

    return (
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
    )
