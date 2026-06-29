from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from apps.runner.acceptance import FakeVerificationSummary, ServicesSummary
from apps.runner.go_live import (
    GoLiveReport,
    build_go_live_report,
    default_go_live_report_path,
    write_go_live_report,
)
from apps.runner.mac_report import RoutingReport
from apps.runner.main import app
from apps.runner.multi_model_report import (
    MultiModelVerificationReport,
    PerModelResult,
    SpecialistSwitchReport,
)
from apps.runner.readiness import ReadinessModel, ReadinessReport
from apps.runner.verify import VerifyCheck
from april_common.settings import load_settings

# --- builders --------------------------------------------------------------


def _models(*, present: bool = True) -> list[ReadinessModel]:
    return [
        ReadinessModel(
            id="april-brain",
            role="brain",
            backend="llama_cpp",
            path_basename="granite3.3-2b-q4_k_m.gguf",
            path_exists=True,
        ),
        ReadinessModel(
            id="april-coding",
            role="coding",
            backend="llama_cpp",
            path_basename="qwen3-1.7b-q8_0.gguf",
            path_exists=present,
        ),
        # Embedding models are not chat models and must not affect the count.
        ReadinessModel(
            id="april-embed",
            role="embedding",
            backend="hashed-token",
            path_basename=None,
            path_exists=False,
        ),
    ]


def _readiness(
    *,
    runtime_backend: str = "llama_cpp",
    llama: bool = True,
    voice_enabled: bool = False,
    models: list[ReadinessModel] | None = None,
    warnings: list[str] | None = None,
    blockers: list[str] | None = None,
    api_token_status: str = "configured",
    runtime_token_status: str = "configured",
) -> ReadinessReport:
    return ReadinessReport(
        generated_at="2026-06-29T00:00:00Z",
        os="Darwin 25",
        cpu_architecture="arm64",
        python_version="3.11.15",
        runtime_backend=runtime_backend,
        runtime_is_fake=runtime_backend != "llama_cpp",
        llama_cpp_python_available=llama,
        environment="development",
        voice_enabled=voice_enabled,
        real_model_preflight_ready=not (blockers or []),
        models=_models() if models is None else models,
        api_token_status=api_token_status,
        runtime_token_status=runtime_token_status,
        blockers=blockers or [],
        warnings=warnings or [],
    )


def _fake(summary: str = "pass") -> FakeVerificationSummary:
    ok = summary == "pass"
    return FakeVerificationSummary(
        ran=True,
        checks_total=6,
        checks_passed=6 if ok else 5,
        checks_failed=0 if ok else 1,
        failures=[] if ok else ["core health: boom"],
        summary=summary,  # type: ignore[arg-type]
    )


def _multi_model(
    *,
    summary: str = "pass",
    real_model_verified: bool = True,
    routing_total: int = 8,
    routing_passed: int = 8,
    routing_fallback: int = 0,
    specialist_success: bool = True,
    models_attempted: int = 2,
    models_passed: int = 2,
    checks_failed: int = 0,
    check_failures: list[str] | None = None,
    routing: RoutingReport | None = None,
) -> MultiModelVerificationReport:
    if routing is None:
        routing = RoutingReport(
            total=routing_total,
            passed=routing_passed,
            accuracy=round(routing_passed / routing_total, 4) if routing_total else 0.0,
            fallback_count=routing_fallback,
        )
    brain = PerModelResult(
        model_id="april-brain",
        role="brain",
        backend="llama_cpp",
        path_basename="granite3.3-2b-q4_k_m.gguf",
        available=True,
        load_success=True,
        chat_success=True,
        streaming_success=True,
        unload_success=True,
        structured_brain_json_success=True,
        routing=routing,
    )
    coding = PerModelResult(
        model_id="april-coding",
        role="coding",
        backend="llama_cpp",
        path_basename="qwen3-1.7b-q8_0.gguf",
        available=True,
        load_success=True,
        chat_success=True,
        streaming_success=True,
        unload_success=True,
        smoke_success=True,
    )
    switch = SpecialistSwitchReport(
        attempted=True,
        brain_loaded=True,
        coding_loaded=True,
        coding_unloaded=True,
        reading_loaded=True,
        reading_unloaded=True,
        brain_usable_after=specialist_success,
    )
    return MultiModelVerificationReport(
        generated_at="2026-06-29T00:00:00Z",
        os="Darwin 25",
        cpu_architecture="arm64",
        python_version="3.11.15",
        runtime_backend="llama_cpp",
        real_model_verified=real_model_verified,
        models=[brain, coding],
        specialist_switch=switch,
        verification_level="all" if summary == "pass" else "none",
        models_attempted=models_attempted,
        models_available=models_attempted,
        models_passed=models_passed,
        checks_failed=checks_failed,
        check_failures=check_failures or [],
        summary=summary,  # type: ignore[arg-type]
    )


def _build(
    *,
    home: Path,
    config_valid: bool = True,
    config_errors: list[str] | None = None,
    readiness: ReadinessReport | None = None,
    fake: FakeVerificationSummary | None = None,
    real_requested: bool = True,
    multi_model: MultiModelVerificationReport | None = "default",  # type: ignore[assignment]
    embedding_provider: str = "runtime-local",
    embedding_model_id: str | None = "april-embed-real",
    services: ServicesSummary | None = None,
) -> GoLiveReport:
    if multi_model == "default":
        multi_model = _multi_model()
    return build_go_live_report(
        home=home,
        config_valid=config_valid,
        config_errors=config_errors or [],
        readiness=readiness or _readiness(),
        fake_verification=fake or _fake(),
        real_requested=real_requested,
        multi_model=multi_model,
        embedding_provider=embedding_provider,
        embedding_model_id=embedding_model_id,
        services=services,
    )


# --- schema / pass ---------------------------------------------------------


def test_go_live_report_schema(tmp_path: Path) -> None:
    report = _build(home=tmp_path)
    assert report.report_type == "go_live"
    assert report.schema_version == 1
    assert report.runtime_backend == "llama_cpp"
    assert report.configured_chat_models_count == 2
    assert report.configured_chat_models_present_count == 2
    assert report.routing_cases_total == 8
    assert report.routing_cases_passed == 8
    assert report.routing_fallback_count == 0
    assert report.specialist_switching_verified is True
    # Round-trips through pydantic validation.
    GoLiveReport.model_validate(report.model_dump())


def test_go_live_mocked_pass_returns_pass(tmp_path: Path) -> None:
    report = _build(home=tmp_path)
    assert report.final_status == "pass"
    assert report.acceptance_level == "real_models"
    assert report.real_model_verified is True
    assert report.real_model_ready is True
    assert any("ready on this Mac" in action for action in report.next_actions)


# --- fail rules ------------------------------------------------------------


def test_go_live_fails_when_backend_is_fake(tmp_path: Path) -> None:
    report = _build(
        home=tmp_path,
        readiness=_readiness(runtime_backend="fake"),
        multi_model=None,
    )
    assert report.final_status == "fail"
    assert any("backend is fake" in blocker for blocker in report.blockers)


def test_go_live_fails_when_llama_cpp_python_missing(tmp_path: Path) -> None:
    report = _build(
        home=tmp_path,
        readiness=_readiness(llama=False),
        multi_model=None,
    )
    assert report.final_status == "fail"
    assert any("llama-cpp-python" in blocker for blocker in report.blockers)
    assert any("pip install" in action for action in report.next_actions)


def test_go_live_fails_when_configured_gguf_missing(tmp_path: Path) -> None:
    report = _build(
        home=tmp_path,
        readiness=_readiness(models=_models(present=False)),
        multi_model=None,
    )
    assert report.final_status == "fail"
    assert any("missing" in blocker.lower() for blocker in report.blockers)


def test_go_live_fails_on_routing_fallback(tmp_path: Path) -> None:
    report = _build(home=tmp_path, multi_model=_multi_model(routing_fallback=2))
    assert report.final_status == "fail"
    assert any("routing fell back" in blocker for blocker in report.blockers)


def test_go_live_fails_when_specialist_switching_fails(tmp_path: Path) -> None:
    report = _build(home=tmp_path, multi_model=_multi_model(specialist_success=False))
    assert report.final_status == "fail"
    assert any("specialist switching" in blocker for blocker in report.blockers)


def test_go_live_fails_when_real_model_run_fails(tmp_path: Path) -> None:
    report = _build(
        home=tmp_path,
        multi_model=_multi_model(summary="fail", real_model_verified=False, checks_failed=1),
    )
    assert report.final_status == "fail"
    assert any("load/chat/stream/unload" in blocker for blocker in report.blockers)


def test_go_live_fails_on_invalid_config(tmp_path: Path) -> None:
    report = _build(
        home=tmp_path,
        config_valid=False,
        config_errors=["broken agents.yaml at /Users/secret/configs/agents.yaml"],
        multi_model=None,
    )
    assert report.final_status == "fail"
    assert any("configuration invalid" in blocker for blocker in report.blockers)


def test_go_live_fails_when_services_fail_to_stop(tmp_path: Path) -> None:
    services = ServicesSummary(
        requested=True, mode="real", startup_status="ok", shutdown_status="failed"
    )
    report = _build(home=tmp_path, services=services)
    assert report.final_status == "fail"
    assert any("services failed" in blocker for blocker in report.blockers)


# --- voice / tokens must NOT fail ------------------------------------------


def test_go_live_does_not_fail_when_voice_disabled(tmp_path: Path) -> None:
    report = _build(home=tmp_path, readiness=_readiness(voice_enabled=False))
    # Voice disabled is opt-in: a skipped/warning note, never a failure.
    assert report.final_status != "fail"
    assert report.final_status == "pass"
    assert report.voice_enabled is False
    assert any("opt-in" in warning.lower() for warning in report.warnings)


def test_go_live_token_warnings_do_not_fail(tmp_path: Path) -> None:
    report = _build(
        home=tmp_path,
        readiness=_readiness(
            warnings=["api/runtime tokens"],
            api_token_status="default-development",
        ),
    )
    # Development tokens are a warning, never a go-live hard failure.
    assert report.final_status == "warning"
    assert any("tokens" in warning.lower() for warning in report.warnings)


def test_go_live_warns_when_real_not_run(tmp_path: Path) -> None:
    report = _build(home=tmp_path, real_requested=False, multi_model=None)
    assert report.final_status == "warning"
    assert any("not requested or could not run" in warning for warning in report.warnings)


def test_go_live_warns_when_embeddings_not_runtime_local(tmp_path: Path) -> None:
    report = _build(home=tmp_path, embedding_provider="hashed-token", embedding_model_id=None)
    assert report.final_status == "warning"
    assert report.readiness.embedding_runtime_local is False


# --- redaction -------------------------------------------------------------


def test_go_live_redacts_absolute_paths(tmp_path: Path) -> None:
    report = _build(
        home=tmp_path,
        config_valid=False,
        config_errors=["bad config at /Users/secret/configs/agents.yaml"],
        readiness=_readiness(warnings=["token leak /Users/secret/.env"]),
        multi_model=_multi_model(
            summary="fail",
            real_model_verified=False,
            checks_failed=1,
            check_failures=["april-brain: failed reading /Users/secret/models/granite.gguf"],
        ),
    )
    blob = report.model_dump_json()
    assert "/Users/secret" not in blob
    # Basenames survive so the report is still actionable.
    assert "agents.yaml" in blob


def test_go_live_single_chat_model_makes_switching_not_applicable(tmp_path: Path) -> None:
    solo = [
        ReadinessModel(
            id="april-brain",
            role="brain",
            backend="llama_cpp",
            path_basename="granite3.3-2b-q4_k_m.gguf",
            path_exists=True,
        )
    ]
    brain_only = MultiModelVerificationReport(
        generated_at="2026-06-29T00:00:00Z",
        os="Darwin 25",
        cpu_architecture="arm64",
        python_version="3.11.15",
        runtime_backend="llama_cpp",
        real_model_verified=True,
        models=[
            PerModelResult(
                model_id="april-brain",
                role="brain",
                backend="llama_cpp",
                available=True,
                load_success=True,
                chat_success=True,
                streaming_success=True,
                unload_success=True,
                structured_brain_json_success=True,
                routing=RoutingReport(total=8, passed=8, accuracy=1.0),
            )
        ],
        specialist_switch=None,
        verification_level="all",
        models_attempted=1,
        models_available=1,
        models_passed=1,
        summary="pass",
    )
    report = _build(home=tmp_path, readiness=_readiness(models=solo), multi_model=brain_only)
    assert report.specialist.applicable is False
    assert report.final_status == "pass"


# --- report IO -------------------------------------------------------------


def test_go_live_default_report_path_under_verification(tmp_path: Path) -> None:
    path = default_go_live_report_path(tmp_path)
    assert path.parent == tmp_path / "data" / "verification"
    assert path.name.startswith("go-live-")
    assert path.suffix == ".json"


def test_write_go_live_report_round_trips(tmp_path: Path) -> None:
    report = _build(home=tmp_path)
    out = tmp_path / "data" / "verification" / "go-live.json"
    written = write_go_live_report(report, out)
    assert written.exists()
    parsed = GoLiveReport.model_validate_json(written.read_text(encoding="utf-8"))
    assert parsed.report_type == "go_live"
    assert parsed.final_status == "pass"


# --- CLI -------------------------------------------------------------------


class FakeManager:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.settings = load_settings(root=home)


def _patch_cli(
    monkeypatch,
    tmp_path: Path,
    *,
    readiness: ReadinessReport,
    multi_model: MultiModelVerificationReport | None,
    config_errors: list[str] | None = None,
) -> dict[str, bool]:
    """Wire the go-live CLI over fully fake primitives and record sensitive calls."""
    called: dict[str, bool] = {"verifier": False, "voice": False, "wake": False, "download": False}
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main.validate_configuration", lambda home: config_errors or [])
    monkeypatch.setattr("apps.runner.main.build_readiness_report", lambda home: readiness)
    monkeypatch.setattr(
        "apps.runner.main.run_fake_verification",
        lambda home: [VerifyCheck(name="core health", ok=True, detail="ok")],
    )

    class _StubVerifier:
        def build_report(self) -> MultiModelVerificationReport | None:
            return multi_model

    def _verifier(home, **kwargs):
        called["verifier"] = True
        return _StubVerifier()

    def _voice(*args, **kwargs):
        called["voice"] = True
        raise AssertionError("go-live must never run live voice verification")

    def _wake(*args, **kwargs):
        called["wake"] = True
        raise AssertionError("go-live must never run wake-word verification")

    def _download(*args, **kwargs):
        called["download"] = True
        raise AssertionError("go-live must never download models")

    monkeypatch.setattr("apps.runner.main.run_all_configured_models_verification", _verifier)
    monkeypatch.setattr("apps.runner.main.run_voice_live_verification", _voice)
    monkeypatch.setattr("apps.runner.main.run_wake_word_live_verification", _wake)
    monkeypatch.setattr("apps.runner.main.run_model_downloads", _download)
    return called


def test_go_live_cli_fails_on_fake_backend(tmp_path: Path, monkeypatch) -> None:
    called = _patch_cli(
        monkeypatch, tmp_path, readiness=_readiness(runtime_backend="fake"), multi_model=None
    )
    result = CliRunner().invoke(app, ["april", "go-live"])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    # A fake-backend preflight must short-circuit before spawning the real verifier.
    assert called["verifier"] is False
    assert called["voice"] is False
    assert called["wake"] is False
    assert called["download"] is False


def test_go_live_cli_no_microphone_or_network_on_pass(tmp_path: Path, monkeypatch) -> None:
    called = _patch_cli(monkeypatch, tmp_path, readiness=_readiness(), multi_model=_multi_model())
    result = CliRunner().invoke(app, ["april", "go-live", "--json"])
    assert result.exit_code == 0, result.output
    assert '"report_type": "go_live"' in result.output
    # The real-model verifier ran; no microphone, wake-word, or download path did.
    assert called["verifier"] is True
    assert called["voice"] is False
    assert called["wake"] is False
    assert called["download"] is False


def test_go_live_cli_writes_redacted_report(tmp_path: Path, monkeypatch) -> None:
    _patch_cli(monkeypatch, tmp_path, readiness=_readiness(), multi_model=_multi_model())
    result = CliRunner().invoke(app, ["april", "go-live", "--write-report"])
    assert result.exit_code == 0, result.output
    reports = list((tmp_path / "data" / "verification").glob("go-live-*.json"))
    assert reports, "expected a go-live report under data/verification"
    text = reports[0].read_text(encoding="utf-8")
    assert '"report_type": "go_live"' in text
    assert "/Users/" not in text
