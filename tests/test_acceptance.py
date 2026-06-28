from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from apps.runner.acceptance import (
    AcceptanceFlagError,
    AcceptanceReport,
    run_acceptance,
    validate_acceptance_flags,
)
from apps.runner.main import app
from apps.runner.multi_model_report import MultiModelVerificationReport
from apps.runner.readiness import ReadinessReport
from apps.runner.service_manager import ServiceInfo, ServiceStatus
from apps.runner.verify import VerifyCheck
from apps.runner.voice_live import VoiceLiveReport
from apps.runner.wake_live import WakeWordLiveReport
from april_common.settings import load_settings


class FakeManager:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.settings = load_settings(root=home)


class FakeServiceManager:
    """Records service start/stop calls without spawning anything."""

    def __init__(self, home: Path, *, initially_ok: bool = False, start_ok: bool = True) -> None:
        self.home = home
        self.settings = load_settings(root=home)
        self.startup_timeout_seconds = 20.0
        self._running = initially_ok
        self._start_ok = start_ok
        self.started: list[bool] = []
        self.stopped = 0

    def _status(self, ok: bool) -> ServiceStatus:
        return ServiceStatus(
            runtime=ServiceInfo(
                name="runtime",
                pid=1 if ok else None,
                running=ok,
                healthy=ok,
                log_path=self.home / "logs" / "runtime.log",
            ),
            api=ServiceInfo(
                name="api",
                pid=2 if ok else None,
                running=ok,
                healthy=ok,
                log_path=self.home / "logs" / "api.log",
            ),
        )

    def status(self) -> ServiceStatus:
        return self._status(self._running)

    def start(self, *, fake_backend: bool = False) -> ServiceStatus:
        self.started.append(fake_backend)
        self._running = self._start_ok
        return self._status(self._start_ok)

    def stop(self) -> ServiceStatus:
        self.stopped += 1
        self._running = False
        return self._status(False)


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
    monkeypatch.setattr("apps.runner.acceptance.validate_configuration", lambda home: errors or [])
    monkeypatch.setattr(
        "apps.runner.acceptance.build_readiness_report", lambda home: readiness or _readiness()
    )
    monkeypatch.setattr(
        "apps.runner.acceptance.run_fake_verification",
        lambda home: (
            fake_checks
            if fake_checks is not None
            else [VerifyCheck(name="runtime health", ok=True, detail="ok")]
        ),
    )


# --- orchestrator ----------------------------------------------------------


def test_acceptance_fake_only_is_warning_not_pass_by_default(tmp_path: Path, monkeypatch) -> None:
    _patch_core(monkeypatch, readiness=_readiness())
    report = run_acceptance(tmp_path)
    # Fake/local sanity must never silently look like full Mac readiness.
    assert report.final_status == "warning"
    assert report.acceptance_level == "fake_sanity"
    assert report.fake_verification.summary == "pass"
    assert report.real_model_verification is None
    assert report.requested["require_real_models"] is False
    assert any("not requested" in action.lower() for action in report.next_actions)
    assert any("sanity" in action.lower() for action in report.next_actions)


def test_acceptance_fake_only_passes_with_allow_sanity_pass(tmp_path: Path, monkeypatch) -> None:
    _patch_core(monkeypatch, readiness=_readiness())
    report = run_acceptance(tmp_path, allow_sanity_pass=True)
    # Only with the explicit opt-in does a clean fake-only run report pass.
    assert report.final_status == "pass"
    assert report.acceptance_level == "fake_sanity"
    assert report.requested["allow_sanity_pass"] is True


def test_acceptance_allow_sanity_pass_still_warns_with_readiness_warnings(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_core(monkeypatch, readiness=_readiness(warnings=["api/runtime tokens"]))
    report = run_acceptance(tmp_path, allow_sanity_pass=True)
    # Sanity pass only applies when nothing is even advisory.
    assert report.final_status == "warning"


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
    # Voice/wake passing cannot claim a voice acceptance level without real models;
    # and without real models the run is still only a warning.
    assert report.acceptance_level == "fake_sanity"
    assert report.final_status == "warning"


def test_acceptance_full_wake_voice_level_requires_real_plus_both_voice(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_core(monkeypatch, readiness=_readiness(runtime_backend="llama_cpp"))
    monkeypatch.setattr(
        "apps.runner.acceptance.run_all_configured_models_verification",
        lambda home, **kwargs: _StubRealVerifier(_multi_report("pass")),
    )
    report = run_acceptance(
        tmp_path,
        require_real_models=True,
        voice_live_runner=lambda: _voice_report("pass"),
        wake_word_live_runner=lambda: _wake_report("pass"),
    )
    assert report.acceptance_level == "full_wake_voice"
    assert report.final_status == "pass"


def test_acceptance_real_models_only_level_without_voice(tmp_path: Path, monkeypatch) -> None:
    _patch_core(monkeypatch, readiness=_readiness(runtime_backend="llama_cpp"))
    monkeypatch.setattr(
        "apps.runner.acceptance.run_all_configured_models_verification",
        lambda home, **kwargs: _StubRealVerifier(_multi_report("pass")),
    )
    report = run_acceptance(tmp_path, require_real_models=True)
    assert report.acceptance_level == "real_models"
    assert report.final_status == "pass"


def test_acceptance_require_real_models_fails_on_fake_backend(tmp_path: Path, monkeypatch) -> None:
    # A fake runtime backend can never satisfy --require-real-models.
    _patch_core(monkeypatch, readiness=_readiness(runtime_backend="fake"))
    report = run_acceptance(tmp_path, require_real_models=True)
    assert report.final_status == "fail"
    assert report.real_model_verification is not None
    assert report.real_model_verification.summary == "fail"
    assert any("backend is fake" in action.lower() for action in report.next_actions)


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


def test_acceptance_cli_json_output_contains_no_full_tokens(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_API_TOKEN", "tok-secret-acceptance-xyz")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(monkeypatch, readiness=_readiness())
    result = CliRunner().invoke(app, ["april", "acceptance", "--json", "--allow-sanity-pass"])
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


# --- flag validation -------------------------------------------------------


def test_validate_acceptance_flags_rejects_fake_services_with_require_real() -> None:
    with pytest.raises(AcceptanceFlagError):
        validate_acceptance_flags(require_real_models=True, start_services=True, fake_services=True)


def test_validate_acceptance_flags_rejects_fake_services_without_start() -> None:
    with pytest.raises(AcceptanceFlagError):
        validate_acceptance_flags(
            require_real_models=False, start_services=False, fake_services=True
        )


def test_validate_acceptance_flags_allows_valid_combos() -> None:
    validate_acceptance_flags(require_real_models=True, start_services=True, fake_services=False)
    validate_acceptance_flags(require_real_models=False, start_services=True, fake_services=True)
    validate_acceptance_flags(require_real_models=False, start_services=False, fake_services=False)


def test_acceptance_cli_rejects_fake_services_with_require_real_models(
    tmp_path: Path, monkeypatch
) -> None:
    manager = FakeServiceManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app, ["april", "acceptance", "--require-real-models", "--fake-services"]
    )
    assert result.exit_code == 1
    assert "Cannot combine --fake-services" in result.output
    # No services were touched by a rejected invocation.
    assert manager.started == []
    assert manager.stopped == 0


# --- service orchestration -------------------------------------------------


def _patch_wake_runner(monkeypatch, summary: str) -> None:
    monkeypatch.setattr("apps.runner.main.collect_voice_doctor", lambda settings: {"status": "ok"})

    async def _fake_wake(**kwargs: object) -> WakeWordLiveReport:
        return _wake_report(summary)

    monkeypatch.setattr("apps.runner.main.run_wake_word_live_verification", _fake_wake)


def test_acceptance_cli_starts_and_stops_services(tmp_path: Path, monkeypatch) -> None:
    manager = FakeServiceManager(tmp_path, initially_ok=False, start_ok=True)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(monkeypatch, readiness=_readiness())
    out = tmp_path / "acc.json"
    result = CliRunner().invoke(
        app,
        ["april", "acceptance", "--start-services", "--fake-services", "--report", str(out)],
    )
    assert result.exit_code == 0, result.output
    # Fake services were started, then stopped because acceptance started them.
    assert manager.started == [True]
    assert manager.stopped == 1
    report = AcceptanceReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.services.requested is True
    assert report.services.mode == "fake"
    assert report.services.started_by_acceptance is True
    assert report.services.startup_status == "ok"
    assert report.services.stopped_after_acceptance is True
    assert report.services.shutdown_status == "stopped"


def test_acceptance_cli_keeps_services_running_with_flag(tmp_path: Path, monkeypatch) -> None:
    manager = FakeServiceManager(tmp_path, initially_ok=False, start_ok=True)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(monkeypatch, readiness=_readiness())
    out = tmp_path / "acc.json"
    result = CliRunner().invoke(
        app,
        [
            "april",
            "acceptance",
            "--start-services",
            "--fake-services",
            "--keep-services-running",
            "--report",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert manager.started == [True]
    assert manager.stopped == 0
    report = AcceptanceReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.services.shutdown_status == "kept_running"


def test_acceptance_cli_stops_services_on_live_check_failure(tmp_path: Path, monkeypatch) -> None:
    manager = FakeServiceManager(tmp_path, initially_ok=False, start_ok=True)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(monkeypatch, readiness=_readiness())
    _patch_wake_runner(monkeypatch, "fail")
    result = CliRunner().invoke(
        app,
        ["april", "acceptance", "--start-services", "--fake-services", "--wake-word-live"],
    )
    # The live check failed (exit 1), but services acceptance started were stopped.
    assert result.exit_code == 1
    assert manager.started == [True]
    assert manager.stopped == 1


def test_acceptance_cli_stops_services_on_exception(tmp_path: Path, monkeypatch) -> None:
    manager = FakeServiceManager(tmp_path, initially_ok=False, start_ok=True)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)

    def _boom(*args: object, **kwargs: object) -> AcceptanceReport:
        raise RuntimeError("verifier exploded")

    monkeypatch.setattr("apps.runner.main.run_acceptance", _boom)
    result = CliRunner().invoke(app, ["april", "acceptance", "--start-services", "--fake-services"])
    assert result.exit_code != 0
    # Services started by acceptance are released even when the run raises.
    assert manager.started == [True]
    assert manager.stopped == 1


def test_acceptance_cli_does_not_stop_already_running_services(tmp_path: Path, monkeypatch) -> None:
    manager = FakeServiceManager(tmp_path, initially_ok=True)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(monkeypatch, readiness=_readiness())
    out = tmp_path / "acc.json"
    result = CliRunner().invoke(
        app, ["april", "acceptance", "--start-services", "--report", str(out)]
    )
    assert result.exit_code == 0, result.output
    # Acceptance did not start them, so it must not stop them.
    assert manager.started == []
    assert manager.stopped == 0
    report = AcceptanceReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.services.startup_status == "already_running"
    assert report.services.shutdown_status == "not_applicable"


def test_acceptance_cli_without_start_services_never_touches_manager(
    tmp_path: Path, monkeypatch
) -> None:
    manager = FakeServiceManager(tmp_path, initially_ok=False)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    _patch_core(monkeypatch, readiness=_readiness())
    result = CliRunner().invoke(app, ["april", "acceptance"])
    assert result.exit_code == 0, result.output
    # Preserve existing behavior: no service orchestration without --start-services.
    assert manager.started == []
    assert manager.stopped == 0


# --- documentation ---------------------------------------------------------


def test_docs_no_longer_mix_fake_with_require_real_models() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in ("docs/macbookpro-acceptance.md", "README.md"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "run april --fake acceptance --require-real-models" not in text
        assert "--fake acceptance" not in text
