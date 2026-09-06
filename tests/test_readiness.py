from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from apps.runner.readiness import ReadinessReport, build_readiness_report
from apps.runner.readiness_inspection import _benchmark_evidence
from april_common.audit import AuditLogger
from april_common.benchmark_evidence import evaluate_benchmark_evidence
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.credentials import CredentialKey, InMemoryCredentialStore
from april_common.hardware_profile import safe_hardware_profile
from april_common.service_health import ServiceHealthResult
from april_common.settings import load_settings
from services.api.production_activation import production_activation_failure_reasons


@pytest.fixture(autouse=True)
def _clear_april_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The builder reads process env; isolate it from any leaked APRIL_* override
    # so temp-config scenarios are deterministic.
    for key in list(os.environ):
        if key.startswith("APRIL_"):
            monkeypatch.delenv(key, raising=False)


def _model_entry(model_id: str, *, path: str, backend: str = "llama_cpp", role: str) -> dict:
    return {
        "id": model_id,
        "name": model_id,
        "path": path,
        "backend": backend,
        "role": role,
        "threads": 4,
        "context_size": 1024,
        "temperature": 0.2,
        "max_output_tokens": 256,
    }


def _write_home(
    home: Path,
    *,
    backend: str = "fake",
    models: dict[str, dict] | None = None,
    voice: dict | None = None,
    memory: dict | None = None,
    extra: dict | None = None,
) -> Path:
    configs = home / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    april: dict = {"environment": "development", "runtime": {"backend": backend}}
    if voice is not None:
        april["voice"] = voice
    if memory is not None:
        april["memory"] = memory
    if extra is not None:
        april.update(extra)
    (configs / "april.yaml").write_text(yaml.safe_dump(april), encoding="utf-8")
    if models is None:
        models = {
            "brain": _model_entry("april-brain", path="models/brain.gguf", role="brain"),
        }
    (configs / "models.yaml").write_text(yaml.safe_dump({"models": models}), encoding="utf-8")
    return home


def _production_store() -> InMemoryCredentialStore:
    store = InMemoryCredentialStore()
    store.set(CredentialKey.API_TOKEN, "prod-api-token-for-readiness")
    store.set(CredentialKey.RUNTIME_TOKEN, "prod-runtime-token-for-readiness")
    return store


def test_fake_backend_without_models_is_not_ready(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    report = build_readiness_report(home)
    assert isinstance(report, ReadinessReport)
    assert report.real_model_ready is False
    assert report.runtime_is_fake is True
    assert "runtime backend" in report.blockers
    assert "configured GGUF model files" in report.blockers
    # Actionable commands only, and the authoritative real-verify command is last.
    if not report.llama_cpp_python_available:
        assert "pip install -e '.[runtime]'" in report.next_actions
    assert any(action.startswith("run april verify") for action in report.next_actions)
    assert not any("verify-wake-live" in action for action in report.next_actions)
    reasoning = next(item for item in report.checks if item.name == "reasoning role readiness")
    semantic = next(item for item in report.checks if item.name == "semantic embedding generation")
    assert reasoning.action == (
        "run april model import --role reasoning --id qwen3-4b-reasoning "
        '--name "Qwen3-4B Q4_K_M" --path /ABSOLUTE/LOCAL/PATH '
        "--sha256 EXPECTED_SHA256"
    )
    assert semantic.action == (
        "run april model import --role embedding --id nomic-embed-text-v1.5 "
        '--name "nomic-embed-text-v1.5 Q8" --path /ABSOLUTE/LOCAL/PATH '
        "--sha256 EXPECTED_SHA256"
    )
    assert report.evidence_boundaries["readiness_implementation"] == "implemented_in_code"
    assert report.evidence_boundaries["core_models"] == "blocked_for_safety"
    assert report.evidence_boundaries["reasoning_role"] == "blocked_for_safety"
    assert report.evidence_boundaries["semantic_embeddings"] == "blocked_for_safety"
    assert report.evidence_boundaries["speaker_verification"] == "optional_unavailable"
    assert report.evidence_boundaries["lora_canary"] == "blocked_for_safety"


def test_invalid_gguf_header_is_a_model_blocker(tmp_path: Path) -> None:
    model_path = tmp_path / "models" / "brain.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"not-a-gguf")
    home = _write_home(tmp_path, backend="llama_cpp")

    report = build_readiness_report(home)

    model = next(item for item in report.models if item.id == "april-brain")
    assert model.path_exists is True
    assert model.artifact_status == "invalid_gguf_header"
    check = next(item for item in report.checks if item.name == "configured GGUF model files")
    assert check.status == "blocker"
    assert "invalid_gguf_header" in check.detail


def test_readiness_reports_disabled_finetuning_and_unevaluated_apple_evidence(
    tmp_path: Path,
) -> None:
    report = build_readiness_report(_write_home(tmp_path))

    assert report.fine_tuning_status == "disabled"
    assert report.production_app_status == "not_evaluated"
    assert report.signing_status == "not_evaluated"
    assert report.notarization_status == "not_evaluated"
    assert report.stapling_status == "not_evaluated"
    assert report.gatekeeper_status == "not_evaluated"
    assert report.apple_release_evidence_status == "not_evaluated"
    fine_tuning = next(item for item in report.checks if item.name == "fine-tuning readiness")
    apple = next(
        item for item in report.checks if item.name == "production app and Apple verification"
    )
    assert fine_tuning.status == "skipped"
    assert apple.status == "skipped"


def test_production_activation_requires_keychain_and_rejects_legacy_plaintext(
    tmp_path: Path,
) -> None:
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        extra={"environment": "production"},
    )
    settings = load_settings(root=home, credential_store=_production_store())

    reasons = production_activation_failure_reasons(
        settings=settings,
        runtime_probe=ServiceHealthResult(True, 200, "ok", "ready"),
        runtime_backend="llama_cpp",
        runtime_simulated=False,
        model_registry={
            "production_model_artifacts_ready": True,
            "required_model_ids": ["brain", "coding", "reading"],
            "reasoning_model_ids": ["reasoning"],
        },
        verified_models={"brain", "coding", "reading", "reasoning", "embedding"},
        embeddings={
            "active_provider": "runtime-local",
            "embedding_model_id": "embedding",
            "fell_back_to_hashed_token": False,
            "reindex_required": False,
            "active_generation": "generation",
        },
        live_flags={"voice_conversation_live_verified": False},
        job_worker_ready=True,
        tool_worker_protocol_ready=True,
        tool_worker_self_check=True,
        audit_chain_status="valid",
        database_integrity=SimpleNamespace(ok=True),
        rollout_state={"status": "disabled"},
        finetuning={"status": "ready"},
        credential_store_selected="legacy-development-default",
        legacy_plaintext_credential_detected=True,
    )

    codes = {reason["code"] for reason in reasons}
    assert "production_keychain_unavailable" in codes
    assert "legacy_plaintext_credentials_detected" in codes


def test_benchmark_evidence_requires_current_configuration_fingerprint(
    tmp_path: Path,
) -> None:
    home = _write_home(tmp_path)
    settings = load_settings(root=home)
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_type": "model_setup_comparison",
        "config_fingerprint": "different-configuration",
        "hardware_profile": safe_hardware_profile(),
        "fixture_set": {"id": "fixture-set"},
        "simulated": False,
        "production_eligible": True,
        "unavailable_measurements": ["thermal_throttling"],
    }
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            "CREATE TABLE background_jobs ("
            "job_type TEXT, status TEXT, completed_at TEXT, result_json TEXT)"
        )
        connection.execute(
            "INSERT INTO background_jobs VALUES (?, ?, ?, ?)",
            ("model_setup_comparison", "succeeded", "2026-07-30T00:00:00Z", json.dumps(report)),
        )

    evidence = _benchmark_evidence(settings)

    assert report["config_fingerprint"] != config_fingerprint_digest(home)
    assert evidence["stale"] is True
    assert evidence["incomplete"] is True
    assert evidence["production_eligible"] is False


def test_simulated_benchmark_is_distinct_from_current_real_evidence() -> None:
    evidence = evaluate_benchmark_evidence(
        {
            "hardware_profile": {"id": "current"},
            "config_fingerprint": "config",
            "fixture_set": {"id": "fixtures"},
            "simulated": True,
            "production_eligible": True,
            "unavailable_measurements": ["thermal_throttling"],
        },
        current_hardware_id="current",
        current_config_fingerprint="config",
    )

    assert evidence["simulated"] is True
    assert evidence["current_hardware"] is True
    assert evidence["production_eligible"] is False


def test_present_model_files_clear_the_gguf_blocker(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "brain.gguf").write_bytes(b"GGUF\x00fake")
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        models={"brain": _model_entry("april-brain", path="models/brain.gguf", role="brain")},
    )
    report = build_readiness_report(home)
    gguf = next(c for c in report.checks if c.name == "configured GGUF model files")
    assert gguf.status == "ok"
    assert "configured GGUF model files" not in report.blockers
    assert report.runtime_is_fake is False
    backend_check = next(c for c in report.checks if c.name == "runtime backend")
    assert backend_check.status == "ok"
    assert report.models[0].path_exists is True
    assert report.models[0].path_basename == "brain.gguf"


def test_missing_model_file_is_a_blocker(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        models={"brain": _model_entry("april-brain", path="models/brain.gguf", role="brain")},
    )
    report = build_readiness_report(home)
    assert "configured GGUF model files" in report.blockers
    assert report.real_model_ready is False
    assert report.models[0].path_exists is False


def test_runtime_local_embedding_model_is_reported(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        memory={
            "embedding_provider": "runtime-local",
            "embedding_model_id": "april-embedding",
        },
    )
    report = build_readiness_report(home)

    check = next(c for c in report.checks if c.name == "runtime-local embedding model")
    assert check.status == "blocker"
    assert "april-embedding" in check.detail
    assert "runtime-local embedding model" in report.blockers


def test_production_readiness_warns_without_runtime_local_embedding_role(
    tmp_path: Path,
) -> None:
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        extra={
            "environment": "production",
            "runtime": {"backend": "llama_cpp"},
        },
    )
    report = build_readiness_report(home, credential_store=_production_store())
    assert "runtime-local embedding hardening" in report.warnings
    assert "embedding-role model registration" in report.warnings


def test_readiness_reports_speaker_gate_daemon_and_sentinel_status(
    tmp_path: Path,
) -> None:
    home = _write_home(tmp_path, backend="fake")
    report = build_readiness_report(home)

    assert report.speaker_gate == "off"
    assert report.speaker_gate_supported is False
    assert report.daemon_status == "stopped"
    assert report.daemon_details_available is False
    assert report.sentinel_live_status == "not_verified"
    assert next(c for c in report.checks if c.name == "speaker gate").status == "skipped"
    assert next(c for c in report.checks if c.name == "daemon detailed status").status == (
        "warning"
    )


def test_readiness_detects_verified_sentinel_report(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    verification = home / "data" / "verification"
    verification.mkdir(parents=True)
    (verification / "wake-live.json").write_text(
        json.dumps(
            {
                "report_type": "wake_word_live",
                "pipeline": "sentinel",
                "evidence_mode": "real_hardware",
                "wake_word_live_verified": True,
                "summary": "pass",
            }
        ),
        encoding="utf-8",
    )

    report = build_readiness_report(home)

    assert report.sentinel_live_status == "verified"
    assert next(c for c in report.checks if c.name == "Sentinel live verification").status == ("ok")


def test_readiness_requires_real_hardware_for_complete_voice_report(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    verification = home / "data" / "verification"
    verification.mkdir(parents=True)
    report_path = verification / "voice-conversation-live.json"
    report_path.write_text(
        json.dumps(
            {
                "report_type": "voice_conversation_live",
                "evidence_mode": "injected_test",
                "voice_conversation_live_verified": True,
            }
        ),
        encoding="utf-8",
    )
    injected = build_readiness_report(home)
    assert injected.voice_conversation_live_status == "failed"

    report_path.write_text(
        json.dumps(
            {
                "report_type": "voice_conversation_live",
                "evidence_mode": "real_hardware",
                "voice_conversation_live_verified": True,
            }
        ),
        encoding="utf-8",
    )
    real = build_readiness_report(home)
    assert real.voice_conversation_live_status == "verified"


def test_default_repo_config_keeps_voice_out_of_blockers(tmp_path: Path) -> None:
    # Guard the *shipped* configs/april.yaml end-to-end: with voice off by default
    # and no voice artifacts present, every voice row is skipped (never a blocker),
    # `run april setup voice` is not pushed, and live voice verification is skipped
    # with the "not requested" message.
    home = tmp_path / "home"
    shutil.copytree(Path.cwd() / "configs", home / "configs")
    report = build_readiness_report(home)
    assert report.conversation_summarization_enabled is True
    assert report.reading_model_registered is True
    assert report.conversation_summarization_degrades_safely is True
    assert report.voice_enabled is False
    voice_checks = [c for c in report.checks if c.name.startswith("voice:")]
    assert voice_checks
    assert all(c.status == "skipped" for c in voice_checks)
    assert all(not name.startswith("voice:") for name in report.blockers)
    assert "run april setup voice" not in report.next_actions
    live = next(c for c in report.checks if c.name == "live voice verification")
    assert live.status == "skipped"
    assert live.detail == "Voice disabled; live verification not requested."


def test_voice_disabled_artifacts_are_skipped_not_blockers(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    report = build_readiness_report(home)
    assert report.voice_enabled is False
    voice_checks = [c for c in report.checks if c.name.startswith("voice:")]
    assert voice_checks
    assert all(c.status == "skipped" for c in voice_checks)
    # Voice never blocks model readiness, and disabled voice is not "ready".
    assert all(not name.startswith("voice:") for name in report.blockers)
    assert report.voice_ready is False


def test_voice_enabled_missing_artifacts_block_voice_only(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        voice={
            "enabled": True,
            "whisper_binary_path": "voice/whisper",
            "whisper_model_path": "voice/whisper.bin",
            "piper_binary_path": "voice/piper",
            "piper_model_path": "voice/piper.onnx",
            "wake_word_model_path": "voice/april.onnx",
        },
    )
    report = build_readiness_report(home)
    assert report.voice_enabled is True
    assert report.voice_ready is False
    voice_blockers = [name for name in report.blockers if name.startswith("voice:")]
    assert len(voice_blockers) == 4
    assert "voice: wake-word model" not in voice_blockers
    assert "voice: wake-word model" in report.warnings
    # A voice-only blocker must not flip real_model_ready on its own.
    model_blockers = [name for name in report.blockers if not name.startswith("voice:")]
    assert "runtime backend" not in model_blockers
    assert "run april setup voice" in report.next_actions
    assert "run april voice verify-live --report data/verification/voice-live.json" in (
        report.next_actions
    )


def test_voice_enabled_without_wake_model_can_be_ptt_preflight_ready(tmp_path: Path) -> None:
    voice_root = tmp_path / "voice"
    voice_root.mkdir()
    for name in ("whisper", "whisper.bin", "piper", "piper.onnx"):
        (voice_root / name).write_bytes(b"asset")
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        voice={
            "enabled": True,
            "whisper_binary_path": "voice/whisper",
            "whisper_model_path": "voice/whisper.bin",
            "piper_binary_path": "voice/piper",
            "piper_model_path": "voice/piper.onnx",
            "wake_word_model_path": None,
        },
    )

    report = build_readiness_report(home)

    assert report.voice_preflight_ready is True
    assert "voice: wake-word model" not in report.blockers
    assert "voice: wake-word model" in report.warnings


def test_default_development_tokens_warn_not_block(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    report = build_readiness_report(home)
    token_check = next(c for c in report.checks if c.name == "api/runtime tokens")
    assert token_check.status == "warning"
    assert "api/runtime tokens" in report.warnings
    assert "api/runtime tokens" not in report.blockers
    assert report.api_token_status == "default-development"
    assert "run april setup tokens" in report.next_actions


def test_blank_voice_paths_report_as_not_configured(tmp_path: Path) -> None:
    # Blank optional voice paths from .env must resolve to None, so readiness shows
    # them as "not configured" rather than as the repo root (the Path(".") bug).
    home = _write_home(tmp_path, backend="fake")
    (home / ".env").write_text(
        "\n".join(
            [
                "APRIL_WHISPER_BINARY_PATH=",
                "APRIL_WHISPER_MODEL_PATH=",
                "APRIL_PIPER_BINARY_PATH=",
                "APRIL_PIPER_MODEL_PATH=",
                "APRIL_WAKE_WORD_MODEL_PATH=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_readiness_report(home)
    assert report.voice_artifacts
    assert all(not artifact.configured for artifact in report.voice_artifacts)
    assert all(not artifact.exists for artifact in report.voice_artifacts)
    # The repo-root basename must never appear as a "configured" voice artifact.
    assert all(artifact.basename is None for artifact in report.voice_artifacts)


def test_placeholder_tokens_warn_and_are_never_printed(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    (home / ".env").write_text(
        "APRIL_API_TOKEN=change-me-local-token\nAPRIL_RUNTIME_TOKEN=change-me-runtime-token\n",
        encoding="utf-8",
    )
    report = build_readiness_report(home)
    token_check = next(c for c in report.checks if c.name == "api/runtime tokens")
    assert token_check.status == "warning"
    assert "api/runtime tokens" in report.warnings
    assert "api/runtime tokens" not in report.blockers
    assert report.api_token_status == "placeholder-insecure"
    assert report.runtime_token_status == "placeholder-insecure"
    assert "run april setup tokens" in report.next_actions
    # The placeholder values must never be printed anywhere in the report.
    blob = json.dumps(report.model_dump())
    assert "change-me-local-token" not in blob
    assert "change-me-runtime-token" not in blob


def test_report_is_json_serialisable_and_redacted(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "brain.gguf").write_bytes(b"GGUF")
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        models={"brain": _model_entry("april-brain", path="models/brain.gguf", role="brain")},
        voice={"enabled": True, "whisper_binary_path": "voice/whisper-bin"},
    )
    report = build_readiness_report(home)
    blob = json.dumps(report.model_dump())
    # JSON output mode round-trips and never leaks absolute paths or token values.
    assert json.loads(blob)
    assert str(tmp_path) not in blob
    assert "local-dev-token" not in blob
    assert "local-dev-runtime-token" not in blob
    # Only basenames / status words survive.
    assert report.models[0].path_basename == "brain.gguf"
    assert report.api_token_status == "default-development"


def test_readiness_reports_durable_workflows_and_optional_fixture_evidence(
    tmp_path: Path,
) -> None:
    report = build_readiness_report(_write_home(tmp_path))
    assert report.model_import_uses_durable_jobs is True
    assert report.memory_reindex_uses_durable_jobs is True
    assert isinstance(report.comparison_fixtures_installed, bool)
    assert report.real_benchmark_evidence_exists is False
    assert report.benchmark_evidence_production_eligible is False
    assert not any(
        "benchmark" in blocker.lower() or "fixture" in blocker.lower()
        for blocker in report.blockers
    )


def test_broken_config_reports_a_single_blocker(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "april.yaml").write_text("environment: development\n", encoding="utf-8")
    (configs / "models.yaml").write_text("models: [not, a, mapping]\n", encoding="utf-8")
    report = build_readiness_report(tmp_path)
    assert report.real_model_ready is False
    assert "model registry" in report.blockers


def test_readiness_accepts_empty_and_anchor_lagged_audit_states(tmp_path: Path) -> None:
    home = _write_home(tmp_path)
    empty = build_readiness_report(home)
    empty_check = _check(empty, "audit chain")
    assert empty.audit_chain_status == "valid"
    assert empty_check is not None
    assert empty_check.status == "ok"
    assert "audit chain" not in empty.blockers

    audit_path = home / "logs" / "audit.jsonl"
    logger = AuditLogger(audit_path)
    logger.write({"event_type": "one"})
    anchor_path = audit_path.with_name("audit.jsonl.anchor")
    anchor_path.unlink()
    lagged = build_readiness_report(home)
    lagged_check = _check(lagged, "audit chain")
    assert lagged.audit_chain_status == "anchor_lagged"
    assert lagged_check is not None
    assert lagged_check.status == "ok"


def test_readiness_distinguishes_corrupt_and_unavailable_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _write_home(tmp_path)
    audit_path = home / "logs" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_bytes(b"{}\n{}\n{}\n")
    corrupt = build_readiness_report(home)
    corrupt_check = _check(corrupt, "audit chain")
    assert corrupt.audit_chain_status == "corrupt"
    assert corrupt_check is not None
    assert corrupt_check.status == "blocker"
    assert "invalid_schema(line 1)" in corrupt_check.detail
    assert "audit chain" in corrupt.blockers
    assert corrupt.real_model_preflight_ready is False

    class UnavailableAudit:
        def verify(self):
            from april_common.audit import AuditVerification

            return AuditVerification(
                status="unavailable",
                valid=False,
                corrupt=False,
                anchor_lagged=False,
                record_count=0,
                terminal_sequence=None,
                terminal_hash=None,
            )

    monkeypatch.setattr(
        "apps.runner.readiness_security.audit_logger_for_settings",
        lambda settings, **kwargs: UnavailableAudit(),
    )
    unavailable = build_readiness_report(home)
    unavailable_check = _check(unavailable, "audit chain")
    assert unavailable.audit_chain_status == "unavailable"
    assert unavailable_check is not None
    assert unavailable_check.status == "blocker"
    assert "audit chain" in unavailable.blockers


def test_large_audit_is_unverified_and_blocks_offline_readiness(tmp_path: Path) -> None:
    home = _write_home(tmp_path)
    audit_path = home / "logs" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    report = build_readiness_report(home)
    check = _check(report, "audit chain")
    assert report.audit_chain_status == "unverified_size_limit"
    assert report.audit_chain_verification_required is True
    assert check is not None
    assert check.status == "blocker"
    assert "unverified" in check.detail


# ---------------------------------------------------------------------------
# v2 production-readiness checks
# ---------------------------------------------------------------------------


def _check(report: ReadinessReport, name: str):
    for check in report.checks:
        if check.name == name:
            return check
    return None


def test_loopback_binding_ok_by_default(tmp_path: Path) -> None:
    report = build_readiness_report(_write_home(tmp_path))
    check = _check(report, "loopback-only binding")
    assert check is not None
    assert check.status == "ok"


def test_non_loopback_binding_is_a_blocker(tmp_path: Path) -> None:
    home = _write_home(tmp_path, extra={"api": {"host": "0.0.0.0"}})
    report = build_readiness_report(home)
    check = _check(report, "loopback-only binding")
    assert check is not None
    assert check.status == "blocker"
    assert "0.0.0.0" in check.detail


def test_hashed_token_embeddings_warn_in_production(tmp_path: Path) -> None:
    # Production config validation requires real tokens before readiness runs.
    home = _write_home(
        tmp_path,
        extra={
            "environment": "production",
            "runtime": {"backend": "fake"},
        },
    )
    report = build_readiness_report(home, credential_store=_production_store())
    check = _check(report, "embedding provider hardening")
    assert check is not None
    assert check.status == "warning"
    # In development the same config produces no such check.
    dev_report = build_readiness_report(_write_home(tmp_path))
    assert _check(dev_report, "embedding provider hardening") is None


def test_wake_enabled_without_onnx_model_is_a_blocker(tmp_path: Path) -> None:
    home = _write_home(tmp_path, extra={"wake": {"enabled": True}})
    report = build_readiness_report(home)
    check = _check(report, "wake-word ONNX model")
    assert check is not None
    assert check.status == "blocker"
    # Wake enabled without a live validation record warns.
    sentinel = _check(report, "Sentinel live verification")
    assert sentinel is not None
    assert sentinel.status == "warning"
    # The optional speaker gate stays explicitly disabled and does not block chat.
    gate = _check(report, "speaker gate")
    assert gate is not None
    assert gate.status == "skipped"


def test_wake_disabled_keeps_sentinel_and_speaker_gate_informational(tmp_path: Path) -> None:
    report = build_readiness_report(_write_home(tmp_path))
    assert _check(report, "wake-word ONNX model") is None
    sentinel = _check(report, "Sentinel live verification")
    assert sentinel is not None
    assert sentinel.status == "skipped"
    gate = _check(report, "speaker gate")
    assert gate is not None
    assert gate.status == "skipped"


def test_evolution_without_scheduler_warns(tmp_path: Path) -> None:
    home = _write_home(tmp_path, extra={"evolution": {"enabled": True}})
    report = build_readiness_report(home)
    check = _check(report, "evolution scheduling")
    assert check is not None
    assert check.status == "warning"


def test_evolution_kill_switch_warns_when_scheduled(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        extra={"evolution": {"enabled": True}, "scheduler": {"enabled": True}},
    )
    kill = home / "data" / "evolution" / "DISABLED"
    kill.parent.mkdir(parents=True, exist_ok=True)
    kill.write_text("disabled\n", encoding="utf-8")
    report = build_readiness_report(home)
    check = _check(report, "evolution kill switch")
    assert check is not None
    assert check.status == "warning"


def test_pending_eval_cases_warn(tmp_path: Path) -> None:
    home = _write_home(tmp_path)
    pending = home / "data" / "evolution" / "evals" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / "abc123.yaml").write_text("case_type: negative_feedback\n", encoding="utf-8")
    report = build_readiness_report(home)
    check = _check(report, "pending eval cases")
    assert check is not None
    assert check.status == "warning"
    assert "1 staged eval case(s)" in check.detail


def test_readiness_mirrors_evolution_status_fields(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        extra={"evolution": {"enabled": True}, "scheduler": {"enabled": True}},
    )
    evolution_dir = home / "data" / "evolution"
    (evolution_dir / "reports").mkdir(parents=True, exist_ok=True)
    (evolution_dir / "reports" / "run.json").write_text("{}", encoding="utf-8")
    (evolution_dir / "DISABLED").write_text("disabled\n", encoding="utf-8")
    pending = evolution_dir / "evals" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / "abc123.yaml").write_text("case_type: negative_feedback\n", encoding="utf-8")
    candidates = evolution_dir / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    (candidates / "coding_agent-0.overlay.txt").write_text("guidance\n", encoding="utf-8")

    report = build_readiness_report(home)

    assert report.evolution_enabled is True
    assert report.scheduler_enabled is True
    assert report.evolution_kill_switch_active is True
    assert report.dreamer_last_report_available is True
    assert report.pending_eval_case_count == 1
    assert report.pending_write_capable_overlay_count == 1
    # The report stays redacted: no absolute path may appear anywhere.
    assert str(tmp_path) not in json.dumps(report.model_dump())


def test_speaker_gate_detail_is_honest_about_enroll(tmp_path: Path) -> None:
    home = _write_home(tmp_path, extra={"wake": {"enabled": True}})
    report = build_readiness_report(home)
    gate = _check(report, "speaker gate")
    assert gate is not None
    assert gate.status == "skipped"
    assert "off" in gate.detail
    assert "never a security boundary" in gate.detail
    assert "Anyone near the microphone can wake APRIL." in gate.detail


def test_soft_speaker_gate_reports_operator_model_blocker(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        extra={"wake": {"enabled": True, "speaker_gate": "soft"}},
    )
    report = build_readiness_report(home)
    gate = _check(report, "speaker gate")
    assert gate is not None
    assert gate.status == "blocker"
    assert "wake.speaker_verifier_model_path" in gate.detail
    assert "scripts/speaker_verifier/README.md" in gate.detail
    assert "degrades to off" in gate.detail


def test_soft_speaker_gate_supported_only_with_model_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "models" / "speaker-model.stub"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"test stub, readiness never loads it")
    home = _write_home(
        tmp_path,
        extra={
            "wake": {
                "enabled": True,
                "speaker_gate": "soft",
                "speaker_verifier_model_path": "models/speaker-model.stub",
            }
        },
    )
    monkeypatch.setattr("services.wake.speaker.onnxruntime_importable", lambda: True)

    supported = build_readiness_report(home)

    assert supported.speaker_gate_supported is True
    assert _check(supported, "speaker gate").status == "blocker"
    assert "fresh real-hardware" in _check(supported, "speaker gate").detail

    monkeypatch.setattr("services.wake.speaker.onnxruntime_importable", lambda: False)
    unsupported = build_readiness_report(home)
    assert unsupported.speaker_gate_supported is False
    assert "onnxruntime" in _check(unsupported, "speaker gate").detail


def test_injected_speaker_report_cannot_satisfy_soft_gate(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        extra={"wake": {"enabled": True, "speaker_gate": "soft"}},
    )
    report_dir = home / "data" / "verification"
    report_dir.mkdir(parents=True)
    settings = load_settings(root=home)
    (report_dir / "speaker-live.json").write_text(
        json.dumps(
            {
                "report_type": "speaker_live",
                "generated_at": "2026-07-30T00:00:00Z",
                "expires_after_days": 365,
                "config_fingerprint": config_fingerprint_digest(settings.home),
                "evidence_mode": "injected_test",
                "speaker_live_verified": True,
            }
        ),
        encoding="utf-8",
    )

    report = build_readiness_report(home)

    assert report.speaker_live_status == "failed"
    assert _check(report, "speaker gate").status == "blocker"
    assert report.evidence_boundaries["speaker_verification"] == "blocked_for_safety"


def test_configured_finetuning_without_reviewed_evidence_is_not_ready(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "trainer"
    evaluator = tmp_path / "evaluator"
    trainer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    evaluator.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    home = _write_home(
        tmp_path,
        extra={
            "finetune": {
                "enabled": True,
                "trainer_executable": str(trainer),
                "evaluator_executable": str(evaluator),
            }
        },
    )

    report = build_readiness_report(home)

    assert report.fine_tuning_status == "awaiting_reviewed_evaluation_data"
    check = _check(report, "fine-tuning readiness")
    assert check.status == "warning"
    assert "reviewed" in check.detail
    assert report.evidence_boundaries["fine_tuning"] == "optional_unavailable"


def test_missing_lora_adapter_is_a_blocker_and_present_adapter_warns(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "brain.gguf").write_bytes(b"GGUF")
    model = _model_entry("april-brain", path="models/brain.gguf", role="brain")
    model["adapter_path"] = "models/adapters/brain-lora.gguf"
    home = _write_home(tmp_path, backend="llama_cpp", models={"brain": model})

    report = build_readiness_report(home)
    check = _check(report, "LoRA adapter: april-brain")
    assert check is not None
    assert check.status == "blocker"
    assert "fails hard" in check.detail

    adapter = tmp_path / "models" / "adapters" / "brain-lora.gguf"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"GGUF")
    report_with_adapter = build_readiness_report(home)
    check_with_adapter = _check(report_with_adapter, "LoRA adapter: april-brain")
    assert check_with_adapter is not None
    assert check_with_adapter.status == "warning"
    assert "unverified until a real adapter is trained and gated" in check_with_adapter.detail


def test_unreviewed_write_capable_overlays_warn(tmp_path: Path) -> None:
    home = _write_home(tmp_path)
    candidates = home / "data" / "evolution" / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    (candidates / "coding_agent-0.overlay.txt").write_text("guidance\n", encoding="utf-8")
    # Non-write-capable candidates do not trigger the warning.
    (candidates / "general_agent-0.overlay.txt").write_text("guidance\n", encoding="utf-8")
    report = build_readiness_report(home)
    check = _check(report, "prompt overlay review")
    assert check is not None
    assert check.status == "warning"
    assert "1 overlay candidate(s)" in check.detail


def test_production_readiness_reports_overlay_eval_blockers_redacted(
    tmp_path: Path,
) -> None:
    home = _write_home(
        tmp_path,
        backend="fake",
        extra={
            "environment": "production",
            "runtime": {"backend": "fake"},
        },
    )
    report_dir = home / "data" / "evolution" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dream.json").write_text(
        json.dumps(
            {
                "run_id": "dream",
                "created_at": "2026-07-01T00:00:00Z",
                "phases": {
                    "examine": {
                        "pending_real_runtime": [
                            {
                                "agent": "coding_agent",
                                "status": "skipped_real_runtime",
                                "reason": f"missing runtime at {tmp_path}/private/model.gguf",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_readiness_report(home, credential_store=_production_store())

    assert report.production_real_runtime_eval_required is True
    assert report.overlay_eval_mode == "deterministic_fixture_plus_real_runtime"
    assert report.pending_real_runtime_overlay_blocker_count == 1
    assert "model.gguf" in report.pending_real_runtime_overlay_blockers[0]
    assert "pending real-runtime overlay blockers" in report.warnings
    gate = _check(report, "prompt overlay eval gate")
    assert gate is not None
    assert gate.status == "blocker"
    assert report.hashed_token_embedding_fallback is True
    assert report.embedding_role_model_registered is False
    blob = json.dumps(report.model_dump())
    assert str(tmp_path) not in blob
    assert "prod-api-token" not in blob
