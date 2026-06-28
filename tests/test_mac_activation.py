from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from apps.runner.acceptance import (
    AcceptanceEnvironment,
    AcceptanceReport,
    FakeVerificationSummary,
    ReadinessSummary,
)
from apps.runner.mac_activation import (
    ActivationFlagError,
    MacActivationReport,
    run_mac_activation,
    validate_activation_flags,
)
from apps.runner.main import app
from april_common.errors import ConfigError
from april_common.settings import load_settings


class FakeManager:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.settings = load_settings(root=home)


def _seed_configs(
    home: Path, *, models: str = "ORIGINAL_MODELS", voice: str = "ORIGINAL_VOICE"
) -> None:
    (home / "configs").mkdir(parents=True, exist_ok=True)
    (home / "configs" / "models.yaml").write_text(models, encoding="utf-8")
    (home / "configs" / "april.yaml").write_text(voice, encoding="utf-8")


def _voice_paths() -> dict[str, Path]:
    return {
        "whisper_binary": Path("/v/whisper"),
        "whisper_model": Path("/v/model.bin"),
        "piper_binary": Path("/v/piper"),
        "piper_model": Path("/v/voice.onnx"),
        "wake_word_model": Path("/v/april.onnx"),
    }


class WritingModelSetup:
    """Stand-in for setup_model_set that writes a marker into models.yaml on apply."""

    def __init__(self, home: Path, *, marker: str = "MUTATED_MODELS") -> None:
        self.home = home
        self.marker = marker
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if kwargs.get("apply"):
            (self.home / "configs" / "models.yaml").write_text(self.marker, encoding="utf-8")
        return {
            "applied": bool(kwargs.get("apply")),
            "backup_basename": "models.yaml.bak-x" if kwargs.get("apply") else None,
            "entries": [{"role": "brain", "source_basename": "brain.gguf"}],
            "next_commands": [],
        }


class FailOnApplyVoiceSetup:
    """Validates OK (apply=False) but raises on apply=True (a later-step failure)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if kwargs.get("apply"):
            raise ConfigError("Voice path does not exist: /Users/secret/voice.onnx")
        return _voice_result()


class FailValidationVoiceSetup:
    """Fails even during validation (apply=False)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        raise ConfigError("Voice path does not exist: voice.onnx")


def _flags(**overrides: bool) -> dict[str, bool]:
    base = {
        "apply": True,
        "dry_run": False,
        "skip_voice": False,
        "enable_voice": False,
        "run_acceptance_after": False,
        "acceptance_voice_live": False,
        "acceptance_wake_word_live": False,
        "start_services": False,
        "fake_services": False,
    }
    base.update(overrides)
    return base


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


# --- transactional apply ---------------------------------------------------


def test_activation_validates_everything_before_writing(tmp_path: Path) -> None:
    _seed_configs(tmp_path)
    model_setup = WritingModelSetup(tmp_path)
    voice_setup = RecordingSetup(_voice_result())
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths=_voice_paths(),
        skip_voice=False,
        apply=True,
        model_setup=model_setup,
        voice_setup=voice_setup,
    )
    assert report.final_status == "applied"
    # Validation (apply=False) precedes the write (apply=True) for both axes.
    assert [call["apply"] for call in model_setup.calls] == [False, True]
    assert [call["apply"] for call in voice_setup.calls] == [False, True]
    assert report.transaction.committed is True
    assert report.transaction.backup_created is True


def test_activation_writes_nothing_if_voice_validation_fails(tmp_path: Path) -> None:
    _seed_configs(tmp_path)
    model_setup = WritingModelSetup(tmp_path)
    voice_setup = FailValidationVoiceSetup()
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths=_voice_paths(),
        skip_voice=False,
        apply=True,
        model_setup=model_setup,
        voice_setup=voice_setup,
    )
    assert report.final_status == "failed"
    # Model apply (apply=True) never ran because validation blocked it.
    assert all(call["apply"] is False for call in model_setup.calls)
    assert (tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8") == "ORIGINAL_MODELS"
    assert report.transaction.backup_created is False
    assert report.transaction.committed is False


def test_activation_rolls_back_model_config_on_later_failure(tmp_path: Path) -> None:
    _seed_configs(tmp_path)
    model_setup = WritingModelSetup(tmp_path)
    voice_setup = FailOnApplyVoiceSetup()
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths=_voice_paths(),
        skip_voice=False,
        apply=True,
        model_setup=model_setup,
        voice_setup=voice_setup,
    )
    assert report.final_status == "failed"
    assert report.transaction.requested is True
    assert report.transaction.backup_created is True
    assert report.transaction.committed is False
    assert report.transaction.rolled_back is True
    assert report.transaction.rollback_status == "restored"
    assert report.transaction.rollback_reason is not None
    # Model config was mutated during apply, then restored to the original.
    assert (tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8") == "ORIGINAL_MODELS"
    # The rollback reason is redacted (no absolute private path).
    assert "/Users/secret" not in report.transaction.rollback_reason


def test_activation_no_rollback_leaves_partial_state(tmp_path: Path) -> None:
    _seed_configs(tmp_path)
    model_setup = WritingModelSetup(tmp_path)
    voice_setup = FailOnApplyVoiceSetup()
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths=_voice_paths(),
        skip_voice=False,
        apply=True,
        no_rollback=True,
        model_setup=model_setup,
        voice_setup=voice_setup,
    )
    assert report.final_status == "failed"
    assert report.transaction.rolled_back is False
    assert report.transaction.rollback_status == "skipped"
    # With --no-rollback the mutated config is intentionally left in place.
    assert (tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8") == "MUTATED_MODELS"


def test_activation_transaction_not_requested_in_dry_run(tmp_path: Path) -> None:
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths={},
        skip_voice=True,
        apply=False,
        model_setup=RecordingSetup(_model_result()),
    )
    assert report.transaction.requested is False
    assert report.transaction.committed is False
    assert report.transaction.rollback_status == "not_applicable"


# --- voice enable ----------------------------------------------------------


def test_activation_enable_voice_passes_enable_true_only_when_supplied(tmp_path: Path) -> None:
    _seed_configs(tmp_path)
    enabled_result = {**_voice_result(), "voice_enabled": True}
    voice_setup = RecordingSetup(enabled_result)
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths=_voice_paths(),
        skip_voice=False,
        apply=True,
        enable_voice=True,
        model_setup=RecordingSetup(_model_result()),
        voice_setup=voice_setup,
    )
    apply_calls = [call for call in voice_setup.calls if call["apply"]]
    assert apply_calls
    assert apply_calls[0]["enable"] is True
    # The validation call never enables voice.
    assert all(call["enable"] is False for call in voice_setup.calls if not call["apply"])
    assert report.voice.enabled_requested is True
    assert report.voice.enabled_after_apply is True


def test_activation_does_not_enable_voice_without_flag(tmp_path: Path) -> None:
    _seed_configs(tmp_path)
    voice_setup = RecordingSetup(_voice_result())
    report = run_mac_activation(
        tmp_path,
        model_paths={"brain": Path("/models/brain.gguf")},
        voice_paths=_voice_paths(),
        skip_voice=False,
        apply=True,
        enable_voice=False,
        model_setup=RecordingSetup(_model_result()),
        voice_setup=voice_setup,
    )
    assert all(call["enable"] is False for call in voice_setup.calls)
    assert report.voice.enabled_requested is False
    assert report.voice.enabled_after_apply is False


# --- flag validation -------------------------------------------------------


def test_flags_enable_voice_rejects_skip_voice() -> None:
    with pytest.raises(ActivationFlagError):
        validate_activation_flags(**_flags(enable_voice=True, skip_voice=True))


def test_flags_live_requires_run_acceptance() -> None:
    with pytest.raises(ActivationFlagError):
        validate_activation_flags(
            **_flags(acceptance_voice_live=True, enable_voice=True, run_acceptance_after=False)
        )


def test_flags_live_rejects_skip_voice() -> None:
    with pytest.raises(ActivationFlagError):
        validate_activation_flags(
            **_flags(
                acceptance_voice_live=True,
                run_acceptance_after=True,
                enable_voice=False,
                skip_voice=True,
            )
        )


def test_flags_live_requires_enable_voice() -> None:
    with pytest.raises(ActivationFlagError):
        validate_activation_flags(
            **_flags(acceptance_wake_word_live=True, run_acceptance_after=True, enable_voice=False)
        )


def test_flags_fake_services_rejects_run_acceptance() -> None:
    with pytest.raises(ActivationFlagError):
        validate_activation_flags(
            **_flags(fake_services=True, start_services=True, run_acceptance_after=True)
        )


def test_flags_fake_services_requires_start_services() -> None:
    with pytest.raises(ActivationFlagError):
        validate_activation_flags(**_flags(fake_services=True, start_services=False))


def test_flags_full_live_combo_is_valid() -> None:
    validate_activation_flags(
        **_flags(
            enable_voice=True,
            run_acceptance_after=True,
            acceptance_voice_live=True,
            acceptance_wake_word_live=True,
            start_services=True,
        )
    )


def test_activation_cli_enable_voice_rejected_with_skip_voice(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app,
        [
            "april",
            "setup",
            "mac-activation",
            "--brain",
            "/m/b.gguf",
            "--skip-voice",
            "--enable-voice",
        ],
    )
    assert result.exit_code == 1


def test_activation_cli_live_acceptance_requires_run_acceptance(
    tmp_path: Path, monkeypatch
) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app,
        [
            "april",
            "setup",
            "mac-activation",
            "--brain",
            "/m/b.gguf",
            "--whisper-binary",
            "/v/w",
            "--whisper-model",
            "/v/m.bin",
            "--piper-binary",
            "/v/p",
            "--piper-model",
            "/v/v.onnx",
            "--enable-voice",
            "--apply",
            "--acceptance-voice-live",
        ],
    )
    assert result.exit_code == 1


def test_docs_contain_new_activation_and_reports_examples() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in ("docs/macbookpro-acceptance.md", "README.md"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "--enable-voice" in text
        assert "--acceptance-voice-live" in text
        assert "--acceptance-wake-word-live" in text
        assert "run april reports" in text
        # The fake/real contradictory command must never reappear.
        assert "--fake acceptance" not in text


def test_activation_cli_runs_live_acceptance_with_orchestration(
    tmp_path: Path, monkeypatch
) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.mac_activation.setup_model_set", RecordingSetup(_model_result())
    )
    monkeypatch.setattr(
        "apps.runner.mac_activation.setup_voice_stack",
        RecordingSetup({**_voice_result(), "voice_enabled": True}),
    )
    captured: dict[str, Any] = {}

    def _fake_orchestrated(**kwargs: Any) -> AcceptanceReport:
        captured.update(kwargs)
        return _fake_acceptance("pass", "full_wake_voice")

    monkeypatch.setattr("apps.runner.main._run_acceptance_with_services", _fake_orchestrated)
    result = CliRunner().invoke(
        app,
        [
            "april",
            "setup",
            "mac-activation",
            "--brain",
            "/m/b.gguf",
            "--whisper-binary",
            "/v/w",
            "--whisper-model",
            "/v/m.bin",
            "--piper-binary",
            "/v/p",
            "--piper-model",
            "/v/v.onnx",
            "--wake-word-model",
            "/v/a.onnx",
            "--enable-voice",
            "--apply",
            "--run-acceptance",
            "--acceptance-voice-live",
            "--acceptance-wake-word-live",
            "--start-services",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["require_real_models"] is True
    assert captured["start_services"] is True
    assert captured["voice_live_runner"] is not None
    assert captured["wake_word_live_runner"] is not None
