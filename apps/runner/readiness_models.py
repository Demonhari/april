from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CheckStatus = Literal["ok", "warning", "blocker", "skipped"]

# Exact, copy-pasteable next commands. None of these are executed here.
_INSTALL_RUNTIME = "pip install -e '.[runtime]'"
_SETUP_MODELS = "run april setup models"
_SETUP_VOICE = "run april setup voice"
_SETUP_TOKENS = "run april setup tokens"
_SETUP_EMBEDDINGS = (
    "run april model import --role embedding --id nomic-embed-text-v1.5 "
    '--name "nomic-embed-text-v1.5 Q8" --path /ABSOLUTE/LOCAL/PATH '
    "--sha256 EXPECTED_SHA256"
)
_IMPORT_REASONING = (
    "run april model import --role reasoning --id qwen3-4b-reasoning "
    '--name "Qwen3-4B Q4_K_M" --path /ABSOLUTE/LOCAL/PATH '
    "--sha256 EXPECTED_SHA256"
)
_IMPORT_EMBEDDING = (
    "run april model import --role embedding --id nomic-embed-text-v1.5 "
    '--name "nomic-embed-text-v1.5 Q8" --path /ABSOLUTE/LOCAL/PATH '
    "--sha256 EXPECTED_SHA256"
)
_VERIFY_REAL = (
    "run april verify --all-configured-models --require-real-model "
    "--report data/verification/mac-readiness.json"
)
_VERIFY_VOICE = "run april voice verify-live --report data/verification/voice-live.json"
_VERIFY_WAKE = "run april voice verify-wake-live --report data/verification/wake-live.json"
_VERIFY_VOICE_CONVERSATION = (
    "run april voice verify-conversation-live "
    "--report data/verification/voice-conversation-live.json"
)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ReadinessCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str
    action: str | None = None


class VoiceArtifact(BaseModel):
    name: str
    configured: bool
    exists: bool
    basename: str | None = None


class ReadinessModel(BaseModel):
    id: str
    role: str
    backend: str
    path_basename: str | None
    path_exists: bool
    artifact_status: str = "unknown"


class ReadinessReport(BaseModel):
    schema_version: int = 1
    generated_at: str
    os: str
    cpu_architecture: str
    python_version: str
    runtime_backend: str
    runtime_is_fake: bool
    llama_cpp_python_available: bool
    environment: str
    voice_enabled: bool
    # Offline readiness never proves a real GGUF or live voice path. These stay
    # false until populated from actual verification reports elsewhere.
    real_model_ready: bool = False
    voice_ready: bool = False
    # Preflight means the local prerequisites appear present; it is still not
    # proof that load/chat/stream/unload or live voice succeeded.
    real_model_preflight_ready: bool = False
    voice_preflight_ready: bool = False
    models: list[ReadinessModel] = Field(default_factory=list)
    voice_artifacts: list[VoiceArtifact] = Field(default_factory=list)
    api_token_status: str = "missing"
    runtime_token_status: str = "missing"
    credential_store_selected: str = "unknown"
    legacy_plaintext_credential_detected: bool = False
    audit_chain_status: str = "unknown"
    database_quick_check: str = "not_run"
    database_foreign_key_consistent: bool = False
    database_wal_state: str = "unknown"
    database_integrity_failures: list[str] = Field(default_factory=list)
    last_successful_backup: dict[str, object] | None = None
    speaker_gate: str = "off"
    speaker_gate_supported: bool = False
    daemon_status: str = "unknown"
    daemon_details_available: bool = False
    sentinel_live_status: str = "not_verified"
    voice_conversation_live_status: str = "not_verified"
    embedding_provider: str = "hashed-token"
    lexical_tokenizer_version: str = "unicode-nfkc-casefold-v1"
    hashed_token_implementation_version: str = "hashed-token-unicode-v2"
    hybrid_retrieval_enabled: bool = True
    runtime_batch_embedding_supported: bool = True
    runtime_batch_embedding_max_items: int = 64
    embedding_role_model_registered: bool = False
    reasoning_role_model_registered: bool = False
    reasoning_falls_back_to_brain: bool = True
    conversation_summarization_enabled: bool = True
    reading_model_registered: bool = False
    router_model_id: str | None = None
    router_aliased_to_brain: bool = True
    dedicated_router_available: bool = False
    router_failure_reason: str | None = None
    conversation_summarization_available: bool = False
    conversation_summarization_degrades_safely: bool = True
    hashed_token_embedding_fallback: bool = False
    lora_adapter_missing_count: int = 0
    adapter_lifecycle_consistent: bool = True
    incomplete_adapter_operation_count: int = 0
    evolution_rollout_status: str = "disabled"
    incomplete_rollout_transition_count: int = 0
    active_canary_count: int = 0
    expired_canary_count: int = 0
    rollout_candidate_unavailable_count: int = 0
    rollout_candidate_hash_mismatch_count: int = 0
    rollout_pointer_database_disagreement_count: int = 0
    rollout_rollback_required_count: int = 0
    lora_canary_supported: bool = False
    overlay_eval_mode: str = "deterministic_fixture"
    production_real_runtime_eval_required: bool = False
    pending_real_runtime_overlay_blocker_count: int = 0
    pending_real_runtime_overlay_blockers: list[str] = Field(default_factory=list)
    fine_tuning_status: str = "disabled"
    production_app_status: str = "not_evaluated"
    signing_status: str = "not_evaluated"
    notarization_status: str = "not_evaluated"
    stapling_status: str = "not_evaluated"
    gatekeeper_status: str = "not_evaluated"
    apple_release_evidence_status: str = "not_evaluated"
    # Dreamer/evolution visibility (file-derived only; readiness stays inert).
    evolution_enabled: bool = False
    evolution_kill_switch_active: bool = False
    scheduler_enabled: bool = False
    dreamer_last_report_available: bool = False
    pending_eval_case_count: int = 0
    pending_write_capable_overlay_count: int = 0
    model_import_uses_durable_jobs: bool = True
    memory_reindex_uses_durable_jobs: bool = True
    last_successful_semantic_reindex: str | None = None
    active_vector_generation: str | None = None
    active_embedding_provider: str | None = None
    active_embedding_model_id: str | None = None
    comparison_fixtures_installed: bool = False
    comparison_fixture_set_version: str | None = None
    comparison_fixture_set_sha256: str | None = None
    real_benchmark_evidence_exists: bool = False
    benchmark_evidence_current_hardware: bool = False
    benchmark_evidence_simulated: bool = False
    benchmark_evidence_stale: bool = False
    benchmark_evidence_incomplete: bool = False
    benchmark_evidence_production_eligible: bool = False
    checks: list[ReadinessCheck] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
