from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from apps.runner.acceptance import (
    AcceptanceEnvironment,
    AcceptanceReport,
    FakeVerificationSummary,
    ReadinessSummary,
)
from apps.runner.mac_activation import MacActivationReport, run_mac_activation
from apps.runner.main import app
from april_common.errors import ConfigError
from april_common.settings import load_settings


class FakeManager:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.settings = load_settings(root=home)


class RecordingSetup:
    """Stands in for setup_model_set / setup_voice_stack; records call kwargs."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        out = dict(self.result)
        out["applied"] = bool(kwargs.get("apply"))
        return out


def _model_result() -> dict[str, Any]:
    return {
        "applied": False,
        "backup_basename": None,
        "entries": [{"role": "brain", "source_basename": "brain.gguf"}],
        "next_commands": [],
    }


def _voice_result() -> dict[str, Any]:
    return {
        "applied": False,
        "voice_enabled": False,
        "artifacts": [
            {"name": "whisper.cpp", "basename": "whisper"},
            {"name": "piper", "basename": "piper"},
        ],
        "warnings": [],
        "next_commands": [],
    }


def _fake_acceptance(final_status: str = "pass", level: str = "real_models") -> AcceptanceReport:
    return AcceptanceReport(
        generated_at="2026-06-28T00:00:00Z",
        environment=AcceptanceEnvironment(
            os="Darwin 25",
            cpu_architecture="arm64",
            python_version="3.11.0",
            deployment="test",
            llama_cpp_python_available=True,
            runtime_is_fake=False,
        ),
        runtime_backend="llama_cpp",
        acceptance_level=level,  # type: ignore[arg-type]
        config_valid=True,
        fake_verification=FakeVerificationSummary(ran=True, summary="pass"),
        readiness=ReadinessSummary(real_model_ready=True, voice_enabled=False, voice_ready=False),
        final_status=final_status,  # type: ignore[arg-type]
    )


# --- orchestrator ----------------------------------------------------------


def test_activation_is_dry_run_by_default(tmp_path: Path) -> None:
    model_setup = RecordingSetup(_model_result())
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf"), "coding": None, "reading": None},
        voice_paths={},
        skip_voice=True,
        model_setup=model_setup,
    )
    assert report.mode == "dry_run"
    assert report.final_status == "validated"
    assert report.models.applied is False
    # The shared validator was invoked in dry-run (no config mutation).
    assert model_setup.calls
    assert model_setup.calls[0]["apply"] is False


def test_activation_does_not_modify_config_without_apply(tmp_path: Path) -> None:
    model_setup = RecordingSetup(_model_result())
    voice_setup = RecordingSetup(_voice_result())
    run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths={
            "whisper_binary": Path("/v/whisper"),
            "whisper_model": Path("/v/model.bin"),
            "piper_binary": Path("/v/piper"),
            "piper_model": Path("/v/voice.onnx"),
        },
        skip_voice=False,
        apply=False,
        model_setup=model_setup,
        voice_setup=voice_setup,
    )
    assert all(call["apply"] is False for call in model_setup.calls)
    assert all(call["apply"] is False for call in voice_setup.calls)


def test_activation_applies_model_and_voice_with_mocked_validators(tmp_path: Path) -> None:
    model_setup = RecordingSetup(_model_result())
    voice_setup = RecordingSetup(_voice_result())
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths={
            "whisper_binary": Path("/v/whisper"),
            "whisper_model": Path("/v/model.bin"),
            "piper_binary": Path("/v/piper"),
            "piper_model": Path("/v/voice.onnx"),
            "wake_word_model": Path("/v/april.onnx"),
        },
        apply=True,
        model_setup=model_setup,
        voice_setup=voice_setup,
    )
    assert report.mode == "apply"
    assert report.final_status == "applied"
    assert report.models.applied is True
    assert report.voice.applied is True
    # Voice is configured but never enabled by the wizard (no surprise enable).
    assert voice_setup.calls[0]["enable"] is False
    assert report.voice.enabled is False


def test_activation_runs_mocked_acceptance_after_apply(tmp_path: Path) -> None:
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths={},
        skip_voice=True,
        apply=True,
        run_acceptance_after=True,
        model_setup=RecordingSetup(_model_result()),
        acceptance_runner=lambda: _fake_acceptance("pass", "real_models"),
    )
    assert report.acceptance.ran is True
    assert report.acceptance.final_status == "pass"
    assert report.acceptance.acceptance_level == "real_models"


def test_activation_acceptance_skipped_without_apply(tmp_path: Path) -> None:
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths={},
        skip_voice=True,
        apply=False,
        run_acceptance_after=True,
        model_setup=RecordingSetup(_model_result()),
        acceptance_runner=lambda: _fake_acceptance(),
    )
    assert report.acceptance.ran is False
    assert report.acceptance.skipped_reason is not None
    assert "apply" in report.acceptance.skipped_reason.lower()


def test_activation_rejects_non_gguf_model_path(tmp_path: Path) -> None:
    model_setup = RecordingSetup(_model_result())
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.bin")},
        voice_paths={},
        skip_voice=True,
        model_setup=model_setup,
    )
    assert report.final_status == "failed"
    assert report.models.error is not None
    assert ".gguf" in report.models.error
    # The shared validator is never reached for an obviously-wrong suffix.
    assert model_setup.calls == []


def test_activation_requires_a_model_path(tmp_path: Path) -> None:
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": None, "coding": None, "reading": None},
        voice_paths={},
        skip_voice=True,
        model_setup=RecordingSetup(_model_result()),
    )
    assert report.final_status == "failed"
    assert report.models.error is not None


def test_activation_requires_voice_paths_unless_skipped(tmp_path: Path) -> None:
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths={},
        skip_voice=False,
        model_setup=RecordingSetup(_model_result()),
    )
    assert report.final_status == "failed"
    assert report.voice.error is not None
    assert report.voice.skipped is False


def test_activation_redacts_config_error_paths(tmp_path: Path) -> None:
    def _raising(**kwargs: Any) -> dict[str, Any]:
        raise ConfigError("Model path does not exist: /Users/secret/models/brain.gguf")

    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths={},
        skip_voice=True,
        model_setup=_raising,
    )
    assert report.final_status == "failed"
    assert report.models.error is not None
    assert "/Users/secret/models" not in report.models.error


# --- CLI -------------------------------------------------------------------


def test_activation_cli_dry_run_by_default(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    model_setup = RecordingSetup(_model_result())
    monkeypatch.setattr("apps.runner.mac_activation.setup_model_set", model_setup)
    result = CliRunner().invoke(
        app,
        ["april", "setup", "mac-activation", "--brain", "/models/brain.gguf", "--skip-voice"],
    )
    assert result.exit_code == 0, result.output
    assert "VALIDATED" in result.output
    assert model_setup.calls[0]["apply"] is False


def test_activation_cli_writes_redacted_report_under_data_verification(
    tmp_path: Path, monkeypatch
) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.mac_activation.setup_model_set", RecordingSetup(_model_result())
    )
    result = CliRunner().invoke(
        app,
        [
            "april",
            "setup",
            "mac-activation",
            "--brain",
            "/models/brain.gguf",
            "--skip-voice",
            "--write-report",
        ],
    )
    assert result.exit_code == 0, result.output
    reports = list((tmp_path / "data" / "verification").glob("mac-activation-*.json"))
    assert reports, "expected a mac-activation report under data/verification"
    parsed = MacActivationReport.model_validate_json(reports[0].read_text(encoding="utf-8"))
    assert parsed.report_type == "mac_activation"
    text = reports[0].read_text(encoding="utf-8")
    assert manager.settings.api.token not in text


def test_activation_cli_apply_runs_mocked_acceptance(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.mac_activation.setup_model_set", RecordingSetup(_model_result())
    )
    monkeypatch.setattr(
        "apps.runner.main.run_acceptance", lambda home, **kwargs: _fake_acceptance()
    )
    result = CliRunner().invoke(
        app,
        [
            "april",
            "setup",
            "mac-activation",
            "--brain",
            "/models/brain.gguf",
            "--skip-voice",
            "--apply",
            "--run-acceptance",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "APPLIED" in result.output
    assert "Acceptance: pass" in result.output


def test_activation_cli_apply_and_dry_run_conflict(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app,
        ["april", "setup", "mac-activation", "--brain", "/m/b.gguf", "--apply", "--dry-run"],
    )
    assert result.exit_code == 1
