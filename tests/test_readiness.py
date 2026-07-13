from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from apps.runner.readiness import ReadinessReport, build_readiness_report


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
            "api": {"token": "prod-api-token-for-test"},
            "runtime": {"backend": "llama_cpp", "token": "prod-runtime-token-for-test"},
        },
    )
    report = build_readiness_report(home)
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
                "wake_word_live_verified": True,
                "summary": "pass",
            }
        ),
        encoding="utf-8",
    )

    report = build_readiness_report(home)

    assert report.sentinel_live_status == "verified"
    assert next(c for c in report.checks if c.name == "Sentinel live verification").status == ("ok")


def test_default_repo_config_keeps_voice_out_of_blockers(tmp_path: Path) -> None:
    # Guard the *shipped* configs/april.yaml end-to-end: with voice off by default
    # and no voice artifacts present, every voice row is skipped (never a blocker),
    # `run april setup voice` is not pushed, and live voice verification is skipped
    # with the "not requested" message.
    home = tmp_path / "home"
    shutil.copytree(Path.cwd() / "configs", home / "configs")
    report = build_readiness_report(home)
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


def test_broken_config_reports_a_single_blocker(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "april.yaml").write_text("environment: development\n", encoding="utf-8")
    (configs / "models.yaml").write_text("models: [not, a, mapping]\n", encoding="utf-8")
    report = build_readiness_report(tmp_path)
    assert report.real_model_ready is False
    assert "model registry" in report.blockers


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
            "api": {"token": "prod-api-token-3f9c2a71b4d8"},
            "runtime": {"backend": "fake", "token": "prod-runtime-token-8e1d5c92aa07"},
        },
    )
    report = build_readiness_report(home)
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
    # The unsupported speaker gate is called out, not silently skipped.
    gate = _check(report, "speaker gate")
    assert gate is not None
    assert gate.status == "warning"


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
    assert gate.status == "warning"
    assert "does not enable soft mode" in gate.detail
    assert "never a security boundary" in gate.detail
    assert "Anyone near the microphone can wake APRIL." in gate.detail


def test_soft_speaker_gate_reports_operator_verifier_blocker(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        extra={"wake": {"enabled": True, "speaker_gate": "soft"}},
    )
    report = build_readiness_report(home)
    gate = _check(report, "speaker gate")
    assert gate is not None
    assert gate.status == "warning"
    assert "SpeakerVerifier" in gate.detail
    assert "degrades to off" in gate.detail


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
            "api": {"token": "prod-api-token-3f9c2a71b4d8"},
            "runtime": {"backend": "fake", "token": "prod-runtime-token-8e1d5c92aa07"},
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

    report = build_readiness_report(home)

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
