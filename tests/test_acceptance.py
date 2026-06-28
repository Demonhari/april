from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from apps.runner.acceptance import AcceptanceReport, run_acceptance
from apps.runner.main import app
from apps.runner.multi_model_report import MultiModelVerificationReport
from apps.runner.readiness import ReadinessReport
from apps.runner.verify import VerifyCheck
from apps.runner.voice_live import VoiceLiveReport
from apps.runner.wake_live import WakeWordLiveReport
from april_common.settings import load_settings


class FakeManager:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.settings = load_settings(root=home)


def _readiness(
    *,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    real_model_ready: bool = True,
    runtime_backend: str = "fake",
    next_actions: list[str] | None = None,
) -> ReadinessReport:
    return ReadinessReport(
        generated_at="2026-06-28T00:00:00Z",
        os="Darwin 25",
        cpu_architecture="arm64",
        python_version="3.11.0",
        runtime_backend=runtime_backend,
        runtime_is_fake=runtime_backend != "llama_cpp",
        llama_cpp_python_available=False,
        environment="test",
        real_model_ready=real_model_ready,
        voice_enabled=False,
        voice_ready=False,
        blockers=blockers or [],
        warnings=warnings or [],
        next_actions=next_actions or [],
    )


def _multi_report(summary: str) -> MultiModelVerificationReport:
    return MultiModelVerificationReport(
        generated_at="2026-06-28T00:00:00Z",
        os="Darwin 25",
        cpu_architecture="arm64",
        python_version="3.11.0",
        runtime_backend="fake",
        summary=summary,  # type: ignore[arg-type]
        verification_level="none",
        models_attempted=0,
        models_available=0,
        models_passed=0,
    )


class _StubRealVerifier:
    def __init__(self, report: MultiModelVerificationReport) -> None:
        self._report = report

    def build_report(self) -> MultiModelVerificationReport:
        return self._report


def _voice_report(summary: str) -> VoiceLiveReport:
    return VoiceLiveReport(
        platform="Darwin 25",
        sounddevice_available=True,
        input_device_count=1,
        output_device_count=1,
        whisper_binary_available=True,
        whisper_model_available=True,
        piper_binary_available=True,
        piper_model_available=True,
        wake_word_model_available=False,
        recording_success=summary == "pass",
        stt_success=summary == "pass",
        tts_success=summary == "pass",
        playback_user_confirmed=summary == "pass",
        summary=summary,
        voice_live_verified=summary == "pass",
    )


def _wake_report(summary: str) -> WakeWordLiveReport:
    ok = summary == "pass"
    return WakeWordLiveReport(
        summary=summary,  # type: ignore[arg-type]
        wake_word_configured=True,
        wake_word_detected=ok,
        recording_success=ok,
        stt_success=ok,
        transcript_length=10 if ok else 0,
        api_success=ok,
        tts_success=ok,
        playback_user_confirmed=ok,
        wake_word_live_verified=ok,
    )


def _patch_core(monkeypatch, *, errors=None, readiness=None, fake_checks=None) -> None:
    monkeypatch.setattr(
        "apps.runner.acceptance.validate_configuration", lambda home: errors or []
    )
    monkeypatch.setattr(
        "apps.runner.acceptance.build_readiness_report", lambda home: readiness or _readiness()
    )
    monkeypatch.setattr(
        "apps.runner.acceptance.run_fake_verification",
        lambda home: fake_checks
        if fake_checks is not None
        else [VerifyCheck(name="runtime health", ok=True, detail="ok")],
    )


# --- orchestrator ----------------------------------------------------------


def test_acceptance_passes_with_fake_only_and_reports_real_not_required(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_core(monkeypatch, readiness=_readiness())
    report = run_acceptance(tmp_path)
    assert report.final_status == "pass"
    assert report.fake_verification.summary == "pass"
    assert report.real_model_verification is None
    assert report.requested["require_real_models"] is False
    assert any("not requested" in action.lower() for action in report.next_actions)


def test_acceptance_warns_when_readiness_has_blockers_but_real_not_required(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_core(monkeypatch, readiness=_readiness(blockers=["runtime backend"]))
    report = run_acceptance(tmp_path)
    # Real-model blockers are advisory when real models were not required.
    assert report.final_status == "warning"


def test_acceptance_fails_when_real_models_required_but_missing(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_core(monkeypatch, readiness=_readiness(blockers=["configured GGUF model files"]))
    monkeypatch.setattr(
        "apps.runner.acceptance.run_all_configured_models_verification",
        lambda home, **kwargs: _StubRealVerifier(_multi_report("fail")),
    )
    report = run_acceptance(tmp_path, require_real_models=True)
    assert report.final_status == "fail"
    assert report.real_model_verification is not None
    assert report.real_model_verification.summary == "fail"
    assert report.requested["require_real_models"] is True


def test_acceptance_config_invalid_skips_fake_and_fails(tmp_path: Path, monkeypatch) -> None:
    _patch_core(monkeypatch, errors=["bad agents.yaml at /Users/secret/configs/agents.yaml"])
    report = run_acceptance(tmp_path)
    assert report.final_status == "fail"
    assert report.config_valid is False
    assert report.fake_verification.ran is False
    assert report.fake_verification.summary == "skipped"
    # Embedded absolute paths in config errors are redacted to basenames.
    assert "/Users/secret/configs" not in " ".join(report.config_errors)


def test_acceptance_fails_when_fake_verification_fails(tmp_path: Path, monkeypatch) -> None:
    _patch_core(
        monkeypatch,
        fake_checks=[VerifyCheck(name="core health", ok=False, detail="boom")],
    )
    report = run_acceptance(tmp_path)
    assert report.final_status == "fail"
    assert report.fake_verification.summary == "fail"
    assert report.fake_verification.failures


def test_acceptance_folds_live_voice_and_wake_runners(tmp_path: Path, monkeypatch) -> None:
    _patch_core(monkeypatch, readiness=_readiness())
    report = run_acceptance(
        tmp_path,
        voice_live_runner=lambda: _voice_report("pass"),
        wake_word_live_runner=lambda: _wake_report("pass"),
    )
    assert report.voice_live is not None
    assert report.voice_live.summary == "pass"
    assert report.wake_word_live is not None
    assert report.wake_word_live.summary == "pass"
    assert report.requested["voice_live"] is True
    assert report.requested["wake_word_live"] is True
    assert report.final_status == "pass"


def test_acceptance_fails_when_requested_wake_word_fails(tmp_path: Path, monkeypatch) -> None:
    _patch_core(monkeypatch, readiness=_readiness())
    report = run_acceptance(tmp_path, wake_word_live_runner=lambda: _wake_report("fail"))
    assert report.final_status == "fail"
    assert report.wake_word_live is not None
    assert report.wake_word_live.summary == "fail"


# --- CLI -------------------------------------------------------------------


def test_acceptance_cli_writes_redacted_report_under_data_verification(
    tmp_path: Path, monkeypatch
) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(monkeypatch, readiness=_readiness())
    result = CliRunner().invoke(app, ["april", "acceptance", "--write-report"])
    assert result.exit_code == 0, result.output
    reports = list((tmp_path / "data" / "verification").glob("acceptance-*.json"))
    assert reports, "expected an acceptance report under data/verification"
    text = reports[0].read_text(encoding="utf-8")
    # The redacted report never carries token values.
    assert manager.settings.api.token not in text
    if manager.settings.runtime.token:
        assert manager.settings.runtime.token not in text


def test_acceptance_cli_json_output_contains_no_full_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("APRIL_API_TOKEN", "tok-secret-acceptance-xyz")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(monkeypatch, readiness=_readiness())
    result = CliRunner().invoke(app, ["april", "acceptance", "--json"])
    assert result.exit_code == 0, result.output
    assert "tok-secret-acceptance-xyz" not in result.output
    assert '"final_status": "pass"' in result.output


def test_acceptance_cli_exits_nonzero_on_fail(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(
        monkeypatch,
        fake_checks=[VerifyCheck(name="core health", ok=False, detail="boom")],
    )
    result = CliRunner().invoke(app, ["april", "acceptance"])
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_acceptance_cli_custom_report_path(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(monkeypatch, readiness=_readiness())
    out = tmp_path / "custom" / "acceptance.json"
    result = CliRunner().invoke(app, ["april", "acceptance", "--report", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    parsed = AcceptanceReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert parsed.report_type == "acceptance"
