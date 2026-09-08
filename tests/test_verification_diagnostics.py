from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import typer
from fastapi import HTTPException
from typer.testing import CliRunner

import apps.runner.commands.common as common_commands
import apps.runner.commands.runner_verification as runner_verification_commands
import apps.runner.multi_model_report as multi_model_report
import apps.runner.verification.services as verification_services
import apps.runner.verify as verify_module
from apps.runner.acceptance import (
    AcceptanceEnvironment,
    AcceptanceReport,
    FakeVerificationSummary,
    ReadinessSummary,
    RealModelSummary,
    VoiceLiveSummary,
)
from apps.runner.commands.runner_verification import (
    _print_routing_summary,
    _routing_counter,
    _routing_counts,
    _routing_failure_categories,
    _routing_provenance_counts,
    _routing_semantic_counts,
    _routing_stage_codes,
    _runtime_process_summary,
)
from apps.runner.mac_report import ReportThresholds, RoutingCaseResult, RoutingReport
from apps.runner.main import app
from apps.runner.multi_model_report import (
    PerModelResult,
    RoutingOnlyVerificationReport,
)
from apps.runner.soak import SoakReport
from apps.runner.verification.local_checks import (
    run_local_sandbox_verification,
    run_local_security_integrity_verification,
)
from apps.runner.verification.multi_model import _routing_error_code, _VerifierRoutingClient
from apps.runner.verification.reports import (
    brain_decision_after_marker,
    chat_result_from_response,
    json_object_candidates,
    latest_brain_decision_marker,
)
from apps.runner.verification.routing_evidence import (
    _downstream_error_code,
    _downstream_runtime_error_code,
    _routing_stage_code,
)
from apps.runner.verification.services import LauncherVerifier
from apps.runner.verification.types import MissingChatResultError, VerifyCheck
from apps.runner.verify import (
    BenchmarkResult,
    TargetMacValidator,
    _git,
    _process_rss_bytes,
    _verification_health_failure,
    run_all_configured_models_verification,
    run_model_benchmark,
    run_real_model_verification,
    run_routing_only_verification,
    run_workflow_verification,
)
from april_common.errors import ConfigError
from services.api import reporting as api_reporting
from services.april_runtime.schemas import ChatMessage, GenerationOptions, ResponseFormat


def test_routing_diagnostics_format_all_failure_and_provenance_counters() -> None:
    cases = [
        RoutingCaseResult(
            id="a",
            ok=False,
            mismatch_codes=["wrong_intent", "wrong_intent"],
            stage_code="contract_rejected",
            coercions=["confidence_defaulted"],
            downstream_error_code="RUNTIME_UNAVAILABLE",
        ),
        RoutingCaseResult(
            id="b",
            ok=True,
            mismatch_codes=["wrong_agent"],
            stage_code="model_generated",
            coercions=["confidence_defaulted", "other"],
            downstream_runtime_error_code="CONTEXT_BUDGET_EXCEEDED",
        ),
    ]
    report = RoutingReport(
        total=2,
        passed=1,
        accuracy=0.5,
        semantic_passed=1,
        semantic_accuracy=0.5,
        deterministic_count=1,
        model_count=1,
        model_repair_count=2,
        fallback_count=3,
        unknown_provenance_count=4,
        inference_failed_count=5,
        cases=cases,
    )

    assert _routing_counts(report) == "1/2 (0.50)"
    assert _routing_semantic_counts(report) == "1/2 (0.50); normalized=1/2"
    assert _routing_provenance_counts(report) == (
        "deterministic=1 model=1 repair=2 fallback=3 unknown=4"
    )
    assert _routing_failure_categories(report) == "wrong_agent=1, wrong_intent=2"
    assert _routing_stage_codes(report) == "contract_rejected=1, model_generated=1"
    assert _routing_counter(report, "coercions") == "confidence_defaulted=2, other=1"
    assert _routing_counter(report, "downstream_error_code") == "RUNTIME_UNAVAILABLE=1"
    assert _routing_counter(report, "downstream_runtime_error_code") == (
        "CONTEXT_BUDGET_EXCEEDED=1"
    )


def test_routing_diagnostics_use_safe_empty_defaults() -> None:
    assert _routing_counts(None) == "0/0 (0.00)"
    assert _routing_semantic_counts(None) == "0/0 (0.00)"
    assert _routing_provenance_counts(None) == (
        "deterministic=0 model=0 repair=0 fallback=0 unknown=0"
    )
    assert _routing_failure_categories(None) == ""
    assert _routing_stage_codes(None) == "none"
    assert _routing_counter(None, "coercions") == "none"
    assert _routing_counter(SimpleNamespace(cases=[]), "coercions") == "none"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("late"), "routing_timeout"),
        (httpx.TimeoutException("late"), "routing_timeout"),
        (httpx.ConnectError("offline"), "routing_connection_error"),
        (OSError("offline"), "routing_connection_error"),
        (RuntimeError("bad payload"), "inference_transport_error"),
    ],
)
def test_routing_error_codes_remain_bounded(error: BaseException, expected: str) -> None:
    assert _routing_error_code(error) == expected


@pytest.mark.asyncio
async def test_verifier_routing_client_preserves_response_diagnostics() -> None:
    calls: list[dict[str, Any]] = []

    class Verifier:
        timeout = 4.0

        def _post_runtime(
            self, path: str, payload: dict[str, Any], *, timeout: float
        ) -> dict[str, Any]:
            calls.append({"path": path, "payload": payload, "timeout": timeout})
            return {
                "request_id": "request-1",
                "model_id": "april-brain",
                "content": '{"intent": "planning"}',
                "usage": {"total_tokens": 2},
            }

    response = await _VerifierRoutingClient(Verifier()).chat(
        model_id="april-brain",
        messages=[ChatMessage(role="user", content="plan")],
        options=GenerationOptions(max_output_tokens=8),
        response_format=ResponseFormat(type="json_object"),
        request_id="request-1",
    )

    assert response.model_id == "april-brain"
    assert response.diagnostics == {"finish_reason_present": False}
    assert calls[0]["path"] == "/runtime/chat"
    assert calls[0]["payload"]["request_id"] == "request-1"
    assert calls[0]["timeout"] == 4.0


def test_runtime_summary_covers_unknown_process_states() -> None:
    def summary(runtime: object) -> str:
        return _runtime_process_summary(SimpleNamespace(runtime_process={"runtime": runtime}))

    assert summary(None) == "unknown process evidence"
    assert summary({"alive": "unknown"}) == "unknown process evidence"
    assert summary({"alive": False, "returncode": -9}) == "exited by signal (returncode=-9)"
    assert summary({"alive": False, "returncode": 1, "signal": "bad"}) == (
        "exited with returncode=1"
    )
    assert summary({"alive": False, "returncode": None, "signal": "SIGTERM"}) == (
        "exited via SIGTERM"
    )
    assert summary({"alive": False, "returncode": object()}) == "unknown process evidence"


def test_print_routing_summary_renders_structured_counters(capsys) -> None:
    report = SimpleNamespace(
        models=[
            SimpleNamespace(
                role="brain",
                routing=RoutingReport(
                    total=2,
                    passed=1,
                    cases=[
                        RoutingCaseResult(id="one", ok=False, mismatch_codes=["wrong_agent"]),
                        RoutingCaseResult(id="two", ok=True, stage_code="model_generated"),
                    ],
                ),
                model_only_routing=None,
                routing_error_code="routing_timeout",
            )
        ],
        runtime_process={"runtime": {"alive": True}},
        runtime_error=False,
        threshold_failures=["below threshold"],
        log_directory_basename=None,
    )

    _print_routing_summary(report)

    output = capsys.readouterr().out
    assert "Brain routing evaluation" in output
    assert "wrong_agent=1" in output
    assert "routing_timeout" in output
    assert "running" in output


class _Manager:
    def __init__(self, home: Path, environment: str = "development") -> None:
        self.home = home
        self.settings = SimpleNamespace(environment=environment)


def test_verify_routing_only_writes_diagnostic_report(tmp_path: Path, monkeypatch) -> None:
    manager = _Manager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.main.run_routing_only_verification",
        lambda home, **_kwargs: RoutingOnlyVerificationReport(
            generated_at="now", model_id="april-brain", backend="llama_cpp", summary="pass"
        ),
    )
    report_path = tmp_path / "routing.json"

    result = CliRunner().invoke(
        app,
        ["april", "verify", "--routing-only", "--json", "--report", str(report_path)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(report_path.read_text(encoding="utf-8"))["report_type"] == "routing_only"
    assert "real_model_exercised" in result.output


def test_verify_real_model_missing_path_is_explicitly_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("APRIL_TEST_GGUF_PATH", raising=False)
    monkeypatch.setattr("apps.runner.main._manager", lambda: _Manager(tmp_path))

    result = CliRunner().invoke(app, ["april", "verify", "--real-model"])

    assert result.exit_code == 0
    assert "Skipping real-model verification" in result.output


def test_verify_real_model_missing_file_fails_without_starting_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.runner.main._manager", lambda: _Manager(tmp_path))

    result = CliRunner().invoke(
        app, ["april", "verify", "--real-model", str(tmp_path / "missing.gguf")]
    )

    assert result.exit_code == 1
    assert "GGUF path does not exist" in result.output


def test_verify_rejects_invalid_adapter_options_and_unsandboxed_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.runner.main._manager", lambda: _Manager(tmp_path))
    result = CliRunner().invoke(
        app,
        ["april", "verify", "--candidate-adapter-model-id", "brain"],
    )
    assert result.exit_code == 1
    assert "only supported" in result.output

    result = CliRunner().invoke(
        app,
        [
            "april",
            "verify",
            "--all-configured-models",
            "--candidate-adapter-model-id",
            "brain",
        ],
    )
    assert result.exit_code == 1
    assert "Use both" in result.output

    invalid_manager = SimpleNamespace(home=tmp_path, settings=SimpleNamespace())
    monkeypatch.setattr("apps.runner.main._manager", lambda: invalid_manager)
    monkeypatch.setattr(
        "apps.runner.commands.runner_verification.load_settings",
        lambda **_kwargs: (_ for _ in ()).throw(ConfigError("bad settings")),
    )
    result = CliRunner().invoke(
        app,
        ["april", "verify", "--fake", "--development-unsandboxed-override"],
    )
    assert result.exit_code == 1
    assert "valid only in development" in result.output


def test_verify_reports_failed_soak_and_routing_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.runner.main._manager", lambda: _Manager(tmp_path))
    monkeypatch.setattr(
        "apps.runner.main.run_fake_soak",
        lambda *_args, **_kwargs: SoakReport(
            generated_at="now", duration_seconds=1, iterations=1, summary="fail"
        ),
    )
    result = CliRunner().invoke(app, ["april", "verify", "--soak"])
    assert result.exit_code == 1
    assert "APRIL Fake Soak Verification" in result.output

    monkeypatch.setattr(
        "apps.runner.main.run_routing_only_verification",
        lambda *_args, **_kwargs: RoutingOnlyVerificationReport(
            generated_at="now", model_id="brain", backend="llama_cpp", summary="fail"
        ),
    )
    result = CliRunner().invoke(app, ["april", "verify", "--routing-only"])
    assert result.exit_code == 1
    assert "routing-only" in result.output


def test_verify_reports_failed_real_model_and_target_mac_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.runner.main._manager", lambda: _Manager(tmp_path))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub")
    monkeypatch.setattr(
        "apps.runner.main.run_real_model_verification",
        lambda *_args, **_kwargs: [SimpleNamespace(ok=False)],
    )
    result = CliRunner().invoke(app, ["april", "verify", "--real-model", str(model)])
    assert result.exit_code == 1

    class FailedValidator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(ok=False)]

    monkeypatch.setattr("apps.runner.main.TargetMacValidator", FailedValidator)
    result = CliRunner().invoke(app, ["april", "verify", "--target-mac"])
    assert result.exit_code == 1


def test_target_mac_validator_handles_missing_runtime_and_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = TargetMacValidator(
        home=tmp_path,
        model_path=None,
        require_real_model=False,
        max_output_tokens=8,
        timeout=1,
    )
    monkeypatch.setattr(
        "apps.runner.verification.target_mac.load_settings",
        lambda **_kwargs: (_ for _ in ()).throw(ConfigError("bad")),
    )
    validator._configuration_load()
    assert validator.settings_error is not None
    assert "bad" in validator.settings_error
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: False)
    assert validator._llama_cpp_import() is False
    assert validator._report_backend() == "unknown"
    validator.checks.append(SimpleNamespace(name="planning route", ok=True))
    assert validator._structured_brain_ok() is True
    validator._configured_gguf_path()
    validator._voice_checks()
    assert validator.checks


def test_target_mac_report_helpers_cover_real_and_manual_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = TargetMacValidator(
        home=tmp_path,
        model_path=tmp_path / "brain.gguf",
        require_real_model=False,
        max_output_tokens=8,
        timeout=1,
    )
    validator.selected_model = tmp_path / "granite-q4_k_m.gguf"
    validator.real_verifier = SimpleNamespace(
        load_time_seconds=1.0,
        first_token_latency_seconds=0.2,
        output_tokens=4,
        tokens_per_second=2.0,
        runtime_rss_bytes=100,
    )
    validator.checks = [
        SimpleNamespace(name="real model load", ok=True),
        SimpleNamespace(name="real model chat", ok=True),
        SimpleNamespace(name="real model stream", ok=True),
        SimpleNamespace(name="real model unload", ok=True),
        SimpleNamespace(name="brain JSON", ok=True),
    ]
    assert validator._report_backend() == "llama_cpp"
    real = validator._real_model_report()
    assert real.attempted is True
    assert real.load_success is True
    monkeypatch.setattr(
        "apps.runner.verification.target_mac.load_settings",
        lambda **_kwargs: SimpleNamespace(runtime=SimpleNamespace(backend="fake")),
    )
    validator.real_verifier = None
    assert validator._report_backend() == "fake"
    monkeypatch.setattr("apps.runner.verification.target_mac.platform.system", lambda: "Darwin")
    monkeypatch.setattr("apps.runner.verification.target_mac.platform.machine", lambda: "mips")
    validator.checks.clear()
    validator._machine_architecture()
    assert validator.checks[-1].ok is False


def test_verify_wrappers_keep_fake_and_routing_orchestration_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Workflow:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self) -> list[str]:
            return ["workflow-ok"]

    monkeypatch.setattr("apps.runner.verify.WorkflowVerifier", Workflow)
    assert run_workflow_verification(tmp_path) == ["workflow-ok"]

    class Multi:
        def __init__(self, **_kwargs: object) -> None:
            self.ran = False

        def run(self) -> None:
            self.ran = True

        def run_routing_only(self) -> str:
            return "routing-only"

    monkeypatch.setattr("apps.runner.verify.AllConfiguredModelsVerifier", Multi)
    assert run_routing_only_verification(tmp_path) == "routing-only"
    verifier = run_all_configured_models_verification(tmp_path)
    assert verifier.ran is True

    monkeypatch.setattr(
        "apps.runner.verify.run_restricted_process_sync",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(RuntimeError, match="Verification Git"):
        _git(tmp_path, "status")


def test_runner_verification_voice_diagnostic_closures(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        runner_verification_commands,
        "_composition_api",
        SimpleNamespace(
            collect_voice_doctor=lambda _settings: {
                "status": "degraded",
                "macos_microphone_permission_guidance": "grant access",
                "wake_word_guidance": "configure model",
            },
            run_voice_live_verification=lambda **kwargs: (
                kwargs["confirm_recording"]("record")
                and kwargs["confirm_transcription"]("transcribe")
                and kwargs["confirm_playback"]("play")
                and (calls.append("voice") or "voice-report")
            ),
            run_wake_word_live_verification=lambda **kwargs: (
                kwargs["confirm_microphone"]("microphone")
                and (calls.append("wake") or "wake-report")
            ),
        ),
    )
    monkeypatch.setattr(runner_verification_commands.asyncio, "run", lambda value: value)
    monkeypatch.setattr(
        runner_verification_commands.typer,
        "confirm",
        lambda _message, default: True,
    )
    assert runner_verification_commands._voice_live_runner("settings")() == "voice-report"
    assert runner_verification_commands._wake_word_live_runner("settings")() == "wake-report"
    assert calls == ["voice", "wake"]


def test_verify_local_security_json_uses_real_check_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.runner.main._manager", lambda: _Manager(tmp_path))
    monkeypatch.setattr(
        "apps.runner.commands.runner_verification.run_local_security_integrity_verification",
        lambda home: [],
    )

    result = CliRunner().invoke(app, ["april", "verify", "--json"])

    assert result.exit_code == 0
    assert '"checks": []' in result.output


def test_verify_wrappers_fail_closed_without_optional_real_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: False)
    missing_real = run_real_model_verification(tmp_path, tmp_path / "model.gguf")
    missing_benchmark = run_model_benchmark(
        tmp_path,
        tmp_path / "model.gguf",
        prompt="hello",
        runs=1,
        max_output_tokens=8,
        keep_loaded=False,
    )
    workflow = run_workflow_verification(tmp_path, real_model=True)

    assert missing_real[0].ok is False
    assert "runtime" in missing_real[0].detail
    assert missing_benchmark[0].ok is False
    assert missing_benchmark[0].run_index == 1
    assert workflow[0].ok is False
    assert "APRIL_TEST_GGUF_PATH" in workflow[0].detail


def test_process_rss_rejects_missing_failed_and_malformed_ps_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _process_rss_bytes(None) is None

    class Completed:
        def __init__(self, returncode: int | None, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    monkeypatch.setattr(
        "apps.runner.verify.run_restricted_process_sync",
        lambda *_args, **_kwargs: Completed(1, "123"),
    )
    assert _process_rss_bytes(42) is None
    monkeypatch.setattr(
        "apps.runner.verify.run_restricted_process_sync",
        lambda *_args, **_kwargs: Completed(0, "not-a-number"),
    )
    assert _process_rss_bytes(42) is None
    monkeypatch.setattr(
        "apps.runner.verify.run_restricted_process_sync",
        lambda *_args, **_kwargs: Completed(0, "12  extra"),
    )
    assert _process_rss_bytes(42) == 12 * 1024


class _Response:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _LauncherClient:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = iter(responses)

    def __enter__(self) -> _LauncherClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, *_args: object, **_kwargs: object) -> _Response:
        return next(self.responses)

    def get(self, *_args: object, **_kwargs: object) -> _Response:
        return next(self.responses)


def _one_response_client(verifier: LauncherVerifier, response: _Response) -> None:
    client = _LauncherClient([response])
    verifier._client = lambda **_kwargs: client


def _bare_launcher(tmp_path: Path) -> LauncherVerifier:
    verifier = object.__new__(LauncherVerifier)
    verifier.temp = tmp_path
    verifier.verify_home = tmp_path / "home"
    verifier.project = tmp_path / "project"
    verifier.second_project = tmp_path / "second"
    verifier.runtime_port = 18001
    verifier.api_port = 18002
    verifier.api_token = "api"
    verifier.runtime_token = "runtime"
    verifier.runtime = None
    verifier.api = None
    verifier.checks = []
    return verifier


def test_launcher_verifier_response_and_failure_paths(tmp_path: Path, monkeypatch) -> None:
    verifier = _bare_launcher(tmp_path)
    client = _LauncherClient(
        [
            _Response({"models": [{"id": "brain"}]}),
            _Response({"id": "project-1"}),
            _Response({"result": {"conversation_id": "one", "status": "ok"}}),
            _Response({"result": {"conversation_id": "two", "status": "ok"}}),
            _Response({"result": {"conversation_id": "three", "status": "ok"}}),
            _Response({"result": {"status": "ok"}}),
            _Response(
                {
                    "result": {
                        "status": "pending_approval",
                        "pending_approval": {
                            "approval_id": "approval-1",
                            "metadata": {"agent_run_id": "run-1"},
                        },
                    }
                }
            ),
        ]
    )
    verifier._client = lambda **_kwargs: client
    assert verifier._check_models() == "1 models"
    assert verifier._register_project() == "project-1"
    assert verifier._multi_turn("project-1") == "one"
    assert verifier._isolated_conversation("project-1", "one") == "three"
    assert verifier._repo_analysis("project-1") == "ok"
    assert verifier._patch_approval("project-1") == "approval-1"

    bad = _bare_launcher(tmp_path)
    bad_client = _LauncherClient([_Response({"models": []})])
    bad._client = lambda **_kwargs: bad_client
    with pytest.raises(RuntimeError, match="no models"):
        bad._check_models()
    bad_client = _LauncherClient(
        [
            _Response({"result": {"conversation_id": "one", "status": "ok"}}),
            _Response({"result": {"status": "not_ok"}}),
        ]
    )
    bad._client = lambda **_kwargs: bad_client
    with pytest.raises(RuntimeError, match="second turn failed"):
        bad._multi_turn("project-1")
    monkeypatch.setattr(
        "apps.runner.verification.services.httpx.get",
        lambda *_args, **_kwargs: _Response({"ready": True, "tool_worker": {}, "jobs": {}}),
    )
    with pytest.raises(RuntimeError, match="Tool Worker"):
        bad._core_readiness()


def test_launcher_shutdown_uses_only_intercepted_process_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 4242

        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

    verifier = _bare_launcher(tmp_path)
    verifier.runtime = Process()
    verifier.api = Process()
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "apps.runner.verification.services.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    verifier._stop()

    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGTERM)]
    assert verifier._services_stopped() == "stopped"


def test_launcher_verifier_rejects_unsafe_response_paths(tmp_path: Path) -> None:
    verifier = _bare_launcher(tmp_path)
    empty_then_ok = _LauncherClient([_Response({}), _Response({"result": {"status": "ok"}})])
    verifier._client = lambda **_kwargs: empty_then_ok
    assert verifier._post_chat_result(
        empty_then_ok, {}, context="retry", retry_missing_result=True
    ) == {"status": "ok"}

    _one_response_client(verifier, _Response({"result": {"conversation_id": "same"}}))
    with pytest.raises(RuntimeError, match="overlapped"):
        verifier._isolated_conversation("project", "same")

    verifier._client = lambda **_kwargs: _LauncherClient(
        [_Response({"id": "second"}), _Response({}, status_code=500)]
    )
    with pytest.raises(RuntimeError, match="expected 403"):
        verifier._conversation_switch_rejected("conversation")
    _one_response_client(verifier, _Response({"result": {"status": "failed"}}))
    with pytest.raises(RuntimeError, match="repo analysis failed"):
        verifier._repo_analysis("project")
    _one_response_client(verifier, _Response({"result": {"status": "ok"}}))
    with pytest.raises(RuntimeError, match="status"):
        verifier._patch_approval("project")
    _one_response_client(
        verifier,
        _Response(
            {
                "result": {
                    "status": "pending_approval",
                    "pending_approval": {"approval_id": "approval", "metadata": {}},
                }
            }
        ),
    )
    with pytest.raises(RuntimeError, match="structured agent"):
        verifier._patch_approval("project")
    _one_response_client(verifier, _Response({"result": {"status": "failed"}}))
    with pytest.raises(RuntimeError, match="failed"):
        verifier._direct_agent_run("project")


def test_launcher_verifier_approval_and_rejection_diagnostics(tmp_path: Path) -> None:
    verifier = _bare_launcher(tmp_path)
    verifier.project.mkdir()
    (verifier.project / "README.md").write_text("unchanged\n", encoding="utf-8")
    (verifier.project / "README.md").write_text("fixed animation\n", encoding="utf-8")

    _one_response_client(
        verifier,
        _Response({"status": "resumed", "result": {"status": "ok"}}),
    )
    assert verifier._approve("approval") == "applied and resumed"
    for payload, message in (
        ({"status": "not-resumed"}, "not-resumed"),
        ({"status": "resumed", "result": {}}, "agent did not"),
    ):
        _one_response_client(verifier, _Response(payload))
        with pytest.raises(RuntimeError, match=message):
            verifier._approve("approval")

    _one_response_client(verifier, _Response({}, status_code=200))
    with pytest.raises(RuntimeError, match="expected 403"):
        verifier._approval_replay_rejected("approval")
    _one_response_client(verifier, _Response({"status": "other"}))
    with pytest.raises(RuntimeError, match="other"):
        verifier._deny_approval("approval")
    _one_response_client(verifier, _Response({}, status_code=500))
    with pytest.raises(RuntimeError, match="expected 200"):
        verifier._deny_approval("approval")
    _one_response_client(verifier, _Response({}, status_code=500))
    with pytest.raises(RuntimeError, match="expected 403"):
        verifier._expired_approval_rejected("approval")


def test_launcher_verifier_rejects_tamper_escape_override_and_bad_cwd(tmp_path: Path) -> None:
    verifier = _bare_launcher(tmp_path)
    verifier.verify_home.mkdir()
    verifier.project.mkdir()
    (verifier.verify_home / "data" / "artifacts" / "patches").mkdir(parents=True)
    request = _Response(
        {
            "approval": {
                "approval_id": "approval",
                "metadata": {"artifact_id": "artifact"},
            }
        }
    )
    client = _LauncherClient([request, _Response({"status": "failed"})])
    verifier._client = lambda **_kwargs: client
    assert verifier._tampered_artifact_rejected("project") == "failed"

    _one_response_client(verifier, _Response({}, status_code=200))
    with pytest.raises(RuntimeError, match="expected 403"):
        verifier._path_escape_rejected("project")
    _one_response_client(verifier, _Response({}, status_code=200))
    with pytest.raises(RuntimeError, match="expected 403"):
        verifier._repo_override_rejected()
    _one_response_client(
        verifier,
        _Response({"approval": {"args": {"cwd": str(verifier.second_project)}}}),
    )
    with pytest.raises(RuntimeError, match="cwd was not forced"):
        verifier._run_command_cwd_forced("project")


def test_launcher_verifier_checks_jobs_streaming_records_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _bare_launcher(tmp_path)
    job_client = _LauncherClient(
        [
            _Response({"id": "job-1"}),
            _Response({"status": "succeeded", "result": {"self_check": True}}),
        ]
    )
    verifier._client = lambda **_kwargs: job_client
    assert verifier._job_self_check() == "submitted, claimed, and durably completed"

    class Stream:
        def __enter__(self) -> Stream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> list[str]:
            return ["event: token", "event: usage"]

    monkeypatch.setattr(
        "apps.runner.verification.services.httpx.stream",
        lambda *_args, **_kwargs: Stream(),
    )
    assert verifier._runtime_streaming() == "1 token events"

    logs = verifier.temp / "logs"
    logs.mkdir()
    (logs / "audit.jsonl").write_text("approved_tool_executed approval_consumed", encoding="utf-8")
    assert verifier._audit_records() == "ok"

    class Row:
        def __init__(self, value: object) -> None:
            self.value = value

        def fetchone(self) -> object:
            return self.value

        def fetchall(self) -> list[tuple[str]]:
            return [("model",)]

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, query: str, *_args: object) -> Row:
            if "tool_calls" in query:
                return Row((1,))
            if "agent_runs" in query and "COUNT" in query:
                return Row((1,))
            if "agent_iterations" in query:
                return Row((1,))
            if "suspended_agent_runs" in query:
                return Row((1,))
            return Row(None)

    monkeypatch.setattr(
        "apps.runner.verification.services.connect_sqlite",
        lambda *_args: Connection(),
    )
    (verifier.temp / "data").mkdir()
    (verifier.temp / "data" / "april.db").write_bytes(b"db")
    assert verifier._tool_call_records() == "1"
    assert verifier._agent_run_records() == "runs=1, iterations=1, suspended=1, route_sources=model"
    assert verifier._suspended_status("approval") == "1"
    assert _bare_launcher(tmp_path / "missing")._suspended_status("approval") is None


def test_launcher_verifier_reports_live_process_and_force_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 4242

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.waits = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["fake"], 5)
            self.returncode = -9
            return -9

    verifier = _bare_launcher(tmp_path)
    process = Process()
    verifier.runtime = process
    verifier.api = None
    signals: list[int] = []
    monkeypatch.setattr(
        "apps.runner.verification.services.os.killpg",
        lambda _pid, sig: signals.append(sig),
    )
    with pytest.raises(RuntimeError, match="still running"):
        verifier._services_stopped()
    verifier._stop()
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_launcher_start_wait_and_readiness_failure_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _bare_launcher(tmp_path)
    verifier.repo_home = tmp_path
    env = {"APRIL_HOME": str(tmp_path)}
    log = tmp_path / "runtime.log"
    monkeypatch.setattr(
        verification_services,
        "build_process_environment",
        lambda *_args, **kwargs: kwargs.get("source", env),
    )
    process = SimpleNamespace(pid=42)
    monkeypatch.setattr(
        verification_services.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    assert verifier._start("services.april_runtime.server", env, log) is process
    assert log.exists()

    monkeypatch.setattr(
        verification_services.time,
        "monotonic",
        iter([0.0, 0.0, 100.0]).__next__,
    )
    monkeypatch.setattr(
        verification_services,
        "probe_service_health",
        lambda *_args, **_kwargs: SimpleNamespace(ok=False, status_code=503, reason="bad"),
    )
    monkeypatch.setattr(verification_services.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="Runtime"):
        verifier._wait_json(verifier.runtime_url)

    monkeypatch.setattr(
        verification_services.httpx,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(httpx.HTTPError("offline")),
    )
    with pytest.raises(RuntimeError, match="could not be read"):
        verifier._core_readiness()
    monkeypatch.setattr(
        verification_services.httpx,
        "get",
        lambda *_args, **_kwargs: _Response({}, status_code=503),
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        verifier._core_readiness()
    monkeypatch.setattr(
        verification_services.httpx,
        "get",
        lambda *_args, **_kwargs: _Response({"ready": True, "tool_worker": {}, "jobs": {}}),
    )
    with pytest.raises(RuntimeError, match="Tool Worker"):
        verifier._core_readiness()


def test_launcher_lifecycle_and_failure_diagnostics_cover_error_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _bare_launcher(tmp_path)
    monkeypatch.setattr(
        verification_services,
        "probe_service_health",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, status_code=200),
    )
    monkeypatch.setattr(verification_services.time, "monotonic", iter([0.0, 1.0]).__next__)
    assert verifier._wait_json(verifier.runtime_url) == {"status": "ok", "http_status": 200}

    monkeypatch.setattr(
        verification_services.httpx,
        "get",
        lambda *_args, **_kwargs: _Response(
            {"ready": False, "failure_reasons": [{"code": "db", "message": "offline"}]}
        ),
    )
    with pytest.raises(RuntimeError, match="db: offline"):
        verifier._core_readiness()
    monkeypatch.setattr(
        verification_services.httpx,
        "get",
        lambda *_args, **_kwargs: _Response(
            {"ready": True, "tool_worker": {"self_check": True}, "jobs": {}}
        ),
    )
    with pytest.raises(RuntimeError, match="Job Worker"):
        verifier._core_readiness()

    monkeypatch.setattr(
        verification_services,
        "probe_service_health",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=False, status_code=None, reason="connection_failed"
        ),
    )
    monkeypatch.setattr(
        verification_services.time,
        "monotonic",
        iter([0.0, 0.0, 100.0]).__next__,
    )
    monkeypatch.setattr(verification_services.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="not reachable"):
        verifier._wait_json(verifier.runtime_url)

    monkeypatch.setattr(verification_services.time, "monotonic", lambda: 0.0)
    verifier._client = lambda **_kwargs: _LauncherClient([_Response({}, status_code=500)])
    with pytest.raises(RuntimeError, match="Job submission"):
        verifier._job_self_check()
    verifier._client = lambda **_kwargs: _LauncherClient(
        [_Response({"id": "job"}), _Response({}, status_code=500)]
    )
    with pytest.raises(RuntimeError, match="Job inspection"):
        verifier._job_self_check()
    verifier._client = lambda **_kwargs: _LauncherClient(
        [_Response({"id": "job"}), _Response({"status": "succeeded", "result": {}})]
    )
    with pytest.raises(RuntimeError, match="malformed"):
        verifier._job_self_check()
    verifier._client = lambda **_kwargs: _LauncherClient(
        [_Response({"id": "job"}), _Response({"status": "failed"})]
    )
    with pytest.raises(RuntimeError, match="ended as failed"):
        verifier._job_self_check()
    verifier._client = lambda **_kwargs: _LauncherClient(
        [_Response({"id": "job"}), _Response({"status": "running"})]
    )
    monkeypatch.setattr(
        verification_services.time,
        "monotonic",
        iter([0.0, 0.0, 100.0]).__next__,
    )
    monkeypatch.setattr(verification_services.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="before timeout"):
        verifier._job_self_check()
    verifier._client = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
    verifier._warm_up_tool_routing("project")

    with pytest.raises(MissingChatResultError):
        verifier._post_chat_result(
            _LauncherClient([_Response({}), _Response({})]),
            {},
            context="final missing result",
            retry_missing_result=True,
        )
    verifier._client = lambda **_kwargs: _LauncherClient(
        [_Response({"id": "second"}), _Response({}, status_code=403)]
    )
    assert verifier._conversation_switch_rejected("conversation") == "403"

    verifier.project.mkdir()
    (verifier.project / "README.md").write_text("unchanged\n", encoding="utf-8")
    verifier._client = lambda **_kwargs: _LauncherClient(
        [_Response({"status": "resumed", "result": {"status": "ok"}})]
    )
    with pytest.raises(RuntimeError, match="patch was not applied"):
        verifier._approve("approval")

    verifier._suspended_status = lambda _approval: "unexpected"
    verifier._client = lambda **_kwargs: _LauncherClient([_Response({"status": "denied"})])
    with pytest.raises(RuntimeError, match="suspended run"):
        verifier._deny_approval("approval")
    verifier._client = lambda **_kwargs: _LauncherClient([_Response({}, status_code=403)])
    with pytest.raises(RuntimeError, match="suspended run"):
        verifier._expired_approval_rejected("approval")

    verifier.verify_home.mkdir()
    artifact_dir = verifier.verify_home / "data" / "artifacts" / "patches"
    artifact_dir.mkdir(parents=True)
    verifier._client = lambda **_kwargs: _LauncherClient(
        [
            _Response({"approval": {"approval_id": "a", "metadata": {"artifact_id": "x"}}}),
            _Response({"status": "unexpected"}),
        ]
    )
    with pytest.raises(RuntimeError, match="unexpected"):
        verifier._tampered_artifact_rejected("project")
    verifier._client = lambda **_kwargs: _LauncherClient([_Response({}, status_code=403)])
    assert verifier._repo_override_rejected() == "403"
    verifier._client = lambda **_kwargs: _LauncherClient(
        [_Response({"approval": {"args": {"cwd": str(verifier.project)}}})]
    )
    assert verifier._run_command_cwd_forced("project") == "forced"

    class BadStream:
        def __enter__(self) -> BadStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> list[str]:
            return []

    monkeypatch.setattr(
        verification_services.httpx,
        "stream",
        lambda *_args, **_kwargs: BadStream(),
    )
    with pytest.raises(RuntimeError, match="tokens=0"):
        verifier._runtime_streaming()
    logs = verifier.temp / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "audit.jsonl").write_text("missing", encoding="utf-8")
    with pytest.raises(RuntimeError, match="audit events"):
        verifier._audit_records()

    class ZeroConnection:
        def __enter__(self) -> ZeroConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, *_args: object) -> SimpleNamespace:
            return SimpleNamespace(fetchone=lambda: (0,), fetchall=lambda: [])

    database = verifier.temp / "data" / "april.db"
    database.parent.mkdir(exist_ok=True)
    database.write_bytes(b"db")
    monkeypatch.setattr(verification_services, "connect_sqlite", lambda *_args: ZeroConnection())
    monkeypatch.setattr(verification_services.time, "monotonic", iter([0.0, 100.0]).__next__)
    with pytest.raises(RuntimeError, match="no tool call"):
        verifier._tool_call_records()
    monkeypatch.setattr(verification_services.time, "monotonic", iter([0.0, 1.0, 100.0]).__next__)
    monkeypatch.setattr(verification_services.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="no tool call"):
        verifier._tool_call_records()
    monkeypatch.setattr(verification_services.time, "monotonic", iter([0.0, 1.0, 100.0]).__next__)
    with pytest.raises(RuntimeError, match="runs=0"):
        verifier._agent_run_records()


def test_target_mac_missing_model_and_host_capability_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = TargetMacValidator(
        home=tmp_path,
        model_path=tmp_path / "missing.gguf",
        require_real_model=True,
        max_output_tokens=8,
        timeout=1,
    )
    assert validator._configured_gguf_path() == tmp_path / "missing.gguf"
    readable_path = tmp_path / "unreadable.gguf"
    readable_path.write_bytes(b"model")
    validator.model_path = readable_path
    monkeypatch.setattr("apps.runner.verification.target_mac.os.access", lambda *_args: False)
    validator._configured_gguf_path()
    assert validator.checks[-1].ok is False
    monkeypatch.setattr("apps.runner.verification.target_mac.platform.system", lambda: "Linux")
    validator._machine_architecture()
    assert validator.checks[-1].status == "manual"
    monkeypatch.setattr(
        "apps.runner.verification.target_mac.sys",
        SimpleNamespace(version_info=SimpleNamespace(major=3, minor=14, micro=0)),
    )
    validator._python_version()
    assert validator.checks[-1].ok is False
    monkeypatch.setattr(
        "apps.runner.evals.run_fake_brain_eval",
        lambda _home: (_ for _ in ()).throw(RuntimeError("eval unavailable")),
    )
    assert validator._routing_report() is None

    no_registry = TargetMacValidator(
        home=tmp_path,
        model_path=None,
        require_real_model=False,
        max_output_tokens=8,
        timeout=1,
    )
    monkeypatch.setattr(
        "apps.runner.verification.target_mac.ModelRegistry.from_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConfigError("invalid registry")),
    )
    assert no_registry._select_model_path() is None
    monkeypatch.setattr(
        "apps.runner.verification.target_mac.os.environ",
        {"APRIL_TEST_GGUF_PATH": str(readable_path)},
    )
    assert no_registry._select_model_path() == readable_path
    monkeypatch.setattr(
        "apps.runner.verification.target_mac.ModelRegistry.from_file",
        lambda *_args, **_kwargs: SimpleNamespace(list=lambda: []),
    )
    monkeypatch.setattr("apps.runner.verification.target_mac.os.environ", {})
    assert no_registry._select_model_path() is None
    monkeypatch.setattr(
        "apps.runner.verification.target_mac.load_settings",
        lambda **_kwargs: SimpleNamespace(voice=SimpleNamespace(enabled=False)),
    )
    monkeypatch.setattr(
        "apps.runner.verification.target_mac.verify_coordinator.query_audio_devices",
        lambda: {"sounddevice_installed": True, "input_devices": [], "output_devices": []},
    )
    monkeypatch.setattr(
        "apps.runner.verification.target_mac.verify_coordinator.voice_doctor",
        lambda _settings: {"components": []},
    )
    no_registry._voice_checks()
    assert any(check.status == "manual" for check in no_registry.checks)


def test_verification_report_and_routing_evidence_reject_malformed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _Response("not-json")
    with pytest.raises(Exception, match="missing result"):
        chat_result_from_response(response, context="chat")
    assert json_object_candidates('```json\n{"ok": true,}\n```') == [{"ok": True}]
    assert json_object_candidates("{bad} }") == []

    database = tmp_path / "events.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE conversation_events (event_type TEXT, payload_json TEXT)")
        connection.executemany(
            "INSERT INTO conversation_events(event_type, payload_json) VALUES (?, ?)",
            [
                ("brain_decision", "not-json"),
                ("brain_decision", "[]"),
                ("brain_decision", '{"request_id": "other"}'),
                ("brain_decision", '{"request_id": "wanted", "conversation_id": "c"}'),
            ],
        )
    assert latest_brain_decision_marker(database) == 4
    assert brain_decision_after_marker(database, 0, request_id="wanted", conversation_id="c") == {
        "request_id": "wanted",
        "conversation_id": "c",
    }
    assert brain_decision_after_marker(database, 0, request_id="never") == {}
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not sqlite")
    assert latest_brain_decision_marker(corrupt) == 0
    assert brain_decision_after_marker(corrupt, 0) == {}
    assert json_object_candidates('{"text":"escaped \\"quote"}') == [{"text": 'escaped "quote'}]
    assert _downstream_error_code(_Response({"error": {"code": "bad"}}, status_code=400)) == "bad"
    assert (
        _downstream_error_code(_Response({"error": {"code": "x" * 100}}, status_code=400))
        == "x" * 64
    )
    assert (
        _downstream_runtime_error_code(
            _Response({"error": {"details": {"error": {"code": "RUNTIME_DOWN"}}}})
        )
        == "RUNTIME_DOWN"
    )
    assert _downstream_runtime_error_code(_Response({"error": {}})) is None
    assert _routing_stage_code({}, _Response({}, status_code=400)) == (
        "missing_correlated_route_event"
    )
    assert (
        _routing_stage_code(
            {"route_source": "fallback", "routing_failure_code": "runtime_unavailable"},
            _Response({}, status_code=500),
        )
        == "runtime_unavailable"
    )
    assert _routing_stage_code({}, _Response({"runtime": "down"}, status_code=503)) == (
        "runtime_unavailable"
    )
    assert _routing_stage_code({"route_source": "model"}, _Response({}, status_code=200)) is None
    assert _routing_stage_code({}, _Response({"error": "down"}, status_code=500)) == (
        "missing_correlated_route_event"
    )

    class InvalidJsonResponse(_Response):
        def json(self) -> object:
            raise ValueError("invalid")

    assert _downstream_runtime_error_code(InvalidJsonResponse({}, status_code=500)) is None
    assert _downstream_runtime_error_code(_Response([], status_code=500)) is None


def test_verify_helpers_cover_health_benchmark_and_port_failure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _verification_health_failure(
        "http://127.0.0.1:1/health", "http://127.0.0.1:1", SimpleNamespace(reason="x")
    ).startswith("Core API")
    assert _verification_health_failure(
        "http://127.0.0.1:2/health",
        "http://127.0.0.1:1",
        SimpleNamespace(reason="authentication_rejected"),
    ).startswith("Runtime authentication")
    assert _verification_health_failure(
        "http://127.0.0.1:2/health",
        "http://127.0.0.1:1",
        SimpleNamespace(reason="endpoint_not_found"),
    ).startswith("Runtime health")

    class Benchmark:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def run(self) -> list[str]:
            return ["benchmark"]

    original_llama_check = verify_module._llama_cpp_installed
    assert isinstance(original_llama_check(), bool)
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: True)
    monkeypatch.setattr("apps.runner.verify.ModelBenchmark", Benchmark)
    assert run_model_benchmark(
        tmp_path,
        tmp_path / "model.gguf",
        prompt="hello",
        runs=1,
        max_output_tokens=8,
        keep_loaded=False,
    ) == ["benchmark"]

    class Socket:
        def __enter__(self) -> Socket:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def bind(self, _address: tuple[str, int]) -> None:
            return None

        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", 43210)

    monkeypatch.setattr(verify_module.socket, "socket", lambda *_args: Socket())
    assert verify_module._free_port() == 43210


def test_acceptance_renderer_includes_real_and_voice_diagnostics(capsys) -> None:
    report = AcceptanceReport(
        generated_at="now",
        environment=AcceptanceEnvironment(
            os="Darwin",
            cpu_architecture="x86_64",
            python_version="3.11",
            deployment="test",
            llama_cpp_python_available=False,
            runtime_is_fake=True,
        ),
        runtime_backend="fake",
        config_valid=True,
        fake_verification=FakeVerificationSummary(ran=True, summary="pass"),
        readiness=ReadinessSummary(
            real_model_ready=False,
            voice_enabled=True,
            voice_ready=False,
        ),
        real_model_verification=RealModelSummary(
            summary="degraded",
            verification_level="none",
            real_model_verified=False,
            models_attempted=1,
            models_available=0,
            models_passed=0,
            checks_failed=1,
        ),
        voice_live=VoiceLiveSummary(
            summary="skipped",
            recording_success=False,
            stt_success=False,
            transcript_length=0,
            tts_success=False,
            playback_user_confirmed=False,
            voice_live_verified=False,
        ),
    )
    runner_verification_commands._print_acceptance(report)
    output = capsys.readouterr().out
    assert "real models" in output
    assert "voice (push-to-talk)" in output


def test_fake_verification_wrapper_combines_launcher_and_sandbox_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Launcher:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self) -> list[VerifyCheck]:
            return [VerifyCheck(name="launcher", ok=True, detail="ok")]

    monkeypatch.setattr("apps.runner.verify.LauncherVerifier", Launcher)
    monkeypatch.setattr(
        "apps.runner.verify.run_local_sandbox_verification",
        lambda _home: [VerifyCheck(name="sandbox", ok=True, detail="ok")],
    )
    checks = verify_module.run_fake_verification(tmp_path)
    assert [check.name for check in checks] == ["launcher", "sandbox"]


def test_common_doctor_renders_optional_worker_states_and_native_desktop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    native_calls: list[tuple[str, str]] = []
    native = SimpleNamespace(
        create_window=lambda title, url, js_api: native_calls.append((title, url)),
        start=lambda: native_calls.append(("start", "")),
    )
    monkeypatch.setitem(sys.modules, "webview", native)
    assert common_commands._open_desktop_native("http://desktop", "token") is True
    assert native_calls == [("APRIL Desktop", "http://desktop"), ("start", "")]

    settings = SimpleNamespace(
        home=tmp_path,
        workers=SimpleNamespace(tool_worker_enabled=True, job_worker_enabled=True),
    )
    manager = SimpleNamespace(home=tmp_path, status=lambda: SimpleNamespace())
    monkeypatch.setattr(common_commands._composition, "_manager", lambda: manager)
    monkeypatch.setattr(common_commands._composition, "_print_status", lambda _status: None)
    monkeypatch.setattr(common_commands, "load_settings", lambda **_kwargs: settings)
    monkeypatch.setattr(
        "services.evolution.adapters.inspect_adapter_state",
        lambda _settings: {"consistent": True},
    )
    monkeypatch.setattr(
        "services.tool_worker.limits.default_tool_worker_runtime_directory",
        lambda _home: tmp_path / "worker",
    )
    monkeypatch.setattr(common_commands.shutil, "which", lambda _name: None)
    monkeypatch.setattr(common_commands, "path_contains_dir", lambda _path: False)
    monkeypatch.setattr(common_commands, "is_april_wrapper", lambda _path: False)
    monkeypatch.setattr(common_commands.os, "access", lambda *_args: False)

    status_path = tmp_path / "data" / "runtime" / "job-worker" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text('{"version": "wrong", "ready": false}', encoding="utf-8")
    monkeypatch.setattr(
        "services.tool_worker.limits.validate_live_socket",
        lambda *_args, **_kwargs: "0600",
    )
    common_commands._doctor()
    assert "status invalid or not ready" in capsys.readouterr().out

    monkeypatch.setattr(
        "services.tool_worker.limits.validate_live_socket",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    status_path.unlink()
    common_commands._doctor()
    assert "not running" in capsys.readouterr().out

    from services.tool_worker.limits import UnsafeToolWorkerSocket

    monkeypatch.setattr(
        "services.tool_worker.limits.validate_live_socket",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(UnsafeToolWorkerSocket("unsafe")),
    )
    common_commands._doctor()
    assert "unsafe socket path" in capsys.readouterr().out


def test_runner_verification_json_and_failure_rendering_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _Manager(tmp_path)

    class Report:
        summary = "pass"
        verification_level = "core"

        def model_dump(self) -> dict[str, str]:
            return {"summary": self.summary, "verification_level": self.verification_level}

    class Verifier:
        def __init__(self) -> None:
            self.checks = [VerifyCheck(name="all", ok=True, detail="ok")]

        def build_report(self) -> Report:
            return Report()

    api = SimpleNamespace(
        _manager=lambda: manager,
        run_all_configured_models_verification=lambda *_args, **_kwargs: Verifier(),
        _print_verification_table=lambda *_args: None,
    )
    monkeypatch.setattr(runner_verification_commands, "_composition_api", api)
    monkeypatch.setattr(
        runner_verification_commands,
        "write_multi_model_report",
        lambda _report, path: path,
    )
    all_report = tmp_path / "all.json"
    result = CliRunner().invoke(
        app,
        ["april", "verify", "--all-configured-models", "--json", "--report", str(all_report)],
    )
    assert result.exit_code == 0, result.output

    class Target:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self) -> list[VerifyCheck]:
            return [VerifyCheck(name="target", ok=True, detail="ok")]

        def build_report(self, **_kwargs: object) -> Report:
            return Report()

    api.TargetMacValidator = Target
    monkeypatch.setattr(runner_verification_commands, "write_report", lambda _report, path: path)
    result = CliRunner().invoke(
        app,
        ["april", "verify", "--target-mac", "--json", "--report", str(tmp_path / "target.json")],
    )
    assert result.exit_code == 0, result.output

    model = tmp_path / "model.gguf"
    model.write_bytes(b"model placeholder")
    api.run_real_model_verification = lambda *_args, **_kwargs: [
        VerifyCheck(name="real", ok=False, detail="failed")
    ]
    result = CliRunner().invoke(app, ["april", "verify", "--real-model", str(model), "--json"])
    assert result.exit_code == 1, result.output

    monkeypatch.setattr(
        runner_verification_commands,
        "run_local_security_integrity_verification",
        lambda _home: [VerifyCheck(name="security", ok=False, detail="failed")],
    )
    result = CliRunner().invoke(app, ["april", "verify", "--json"])
    assert result.exit_code == 1, result.output

    api.run_fake_verification = lambda *_args, **_kwargs: [
        VerifyCheck(name="fake", ok=True, detail="ok")
    ]
    result = CliRunner().invoke(
        app,
        ["april", "verify", "--fake", "--development-unsandboxed-override", "--json"],
    )
    assert result.exit_code == 0, result.output

    api.run_fake_soak = lambda *_args, **_kwargs: SoakReport(
        generated_at="now", duration_seconds=1, iterations=1, summary="pass"
    )
    monkeypatch.setattr(
        runner_verification_commands,
        "write_soak_report",
        lambda _report, path: path,
    )
    result = CliRunner().invoke(
        app,
        ["april", "verify", "--soak", "--json", "--report", str(tmp_path / "soak.json")],
    )
    assert result.exit_code == 0, result.output

    monkeypatch.setattr(
        runner_verification_commands,
        "run_local_security_integrity_verification",
        lambda _home: [VerifyCheck(name="security", ok=True, detail="ok")],
    )
    result = CliRunner().invoke(app, ["april", "verify"])
    assert result.exit_code == 0, result.output

    class FailedVerifier(Verifier):
        def __init__(self) -> None:
            self.checks = [VerifyCheck(name="all", ok=False, detail="failed")]

    api.run_all_configured_models_verification = lambda *_args, **_kwargs: FailedVerifier()
    result = CliRunner().invoke(app, ["april", "verify", "--all-configured-models", "--json"])
    assert result.exit_code == 1, result.output

    class FailedTarget(Target):
        def run(self) -> list[VerifyCheck]:
            return [VerifyCheck(name="target", ok=False, detail="failed")]

    api.TargetMacValidator = FailedTarget
    result = CliRunner().invoke(app, ["april", "verify", "--target-mac", "--json"])
    assert result.exit_code == 1, result.output


def test_multi_model_report_acceptance_and_threshold_failures_are_explicit() -> None:
    thresholds = ReportThresholds(
        min_tokens_per_second=1.0,
        max_load_seconds=1.0,
        max_first_token_latency_seconds=1.0,
        max_rss_mb=1.0,
        min_routing_accuracy=0.90,
        min_model_only_routing_accuracy=0.75,
    )
    brain = PerModelResult(
        model_id="brain",
        role="brain",
        backend="llama_cpp",
        available=True,
        load_success=True,
        chat_success=True,
        streaming_success=True,
        unload_success=True,
        structured_brain_json_success=False,
        failure_detail="prompt token authorization secret",
    )
    specialist = PerModelResult(
        model_id="coding",
        role="coding",
        backend="llama_cpp",
        available=True,
        load_success=True,
        chat_success=True,
        streaming_success=True,
        unload_success=True,
        smoke_success=True,
        smoke_schema_valid=False,
        failure_detail="unsafe absolute path",
    )
    assert "structured Brain JSON check failed" in " ".join(brain.acceptance_failures(thresholds))
    assert "schema check failed" in " ".join(specialist.acceptance_failures(thresholds))
    assert "unsafe absolute path" in " ".join(specialist.acceptance_failures(thresholds))

    missing_axes = brain.model_copy(
        update={
            "routing_evaluation_required": True,
            "routing_error_code": "routing_timeout",
            "model_only_routing": RoutingReport(total=0),
            "tokens_per_second": 2.0,
        }
    )
    failures = multi_model_report.per_model_threshold_failures(missing_axes, thresholds)
    assert any("routing report missing" in item for item in failures)
    assert any("model-only routing report has zero cases" in item for item in failures)

    weak_routing = RoutingReport(
        total=1,
        passed=0,
        accuracy=0.0,
        case_set_complete=True,
        provenance_verified=True,
    )
    weak = missing_axes.model_copy(
        update={
            "routing_error_code": None,
            "routing": weak_routing,
            "model_only_routing": weak_routing,
            "load_duration_seconds": 2.0,
            "first_token_latency_seconds": 2.0,
            "process_rss_bytes": 2 * 1024 * 1024,
            "tokens_per_second": 0.5,
        }
    )
    failures = multi_model_report.per_model_threshold_failures(weak, thresholds)
    assert any("below minimum" in item for item in failures)
    assert any("above maximum" in item for item in failures)

    assert (
        multi_model_report._summary(
            attempted=False,
            real_model_verified=False,
            simulated=False,
            checks_failed=0,
            runtime_error=False,
            require_real_model=True,
            threshold_failures_present=False,
            optional_skipped=False,
            switch_ok=True,
        )
        == "fail"
    )
    assert (
        multi_model_report._summary(
            attempted=True,
            real_model_verified=False,
            simulated=True,
            checks_failed=0,
            runtime_error=False,
            require_real_model=False,
            threshold_failures_present=False,
            optional_skipped=False,
            switch_ok=True,
        )
        == "degraded"
    )
    assert (
        multi_model_report._summary(
            attempted=True,
            real_model_verified=True,
            simulated=False,
            checks_failed=0,
            runtime_error=False,
            require_real_model=True,
            threshold_failures_present=True,
            optional_skipped=False,
            switch_ok=True,
        )
        == "fail"
    )
    assert (
        multi_model_report._summary(
            attempted=True,
            real_model_verified=True,
            simulated=False,
            checks_failed=0,
            runtime_error=False,
            require_real_model=False,
            threshold_failures_present=False,
            optional_skipped=True,
            switch_ok=False,
        )
        == "degraded"
    )

    assert (
        multi_model_report._routing_required_ok(
            weak.model_copy(update={"routing_evaluation_required": False}), thresholds
        )
        is True
    )
    assert multi_model_report._routing_required_ok(weak, thresholds) is False
    zero_axes = missing_axes.model_copy(
        update={
            "routing_error_code": None,
            "routing": RoutingReport(total=0),
            "model_only_routing": None,
        }
    )
    zero_failures = multi_model_report.per_model_threshold_failures(zero_axes, thresholds)
    assert any("routing report has zero cases" in item for item in zero_failures)
    assert any("model-only routing report missing" in item for item in zero_failures)
    assert (
        multi_model_report._routing_required_ok(
            weak.model_copy(update={"routing_error_code": "routing_timeout"}), thresholds
        )
        is False
    )
    assert multi_model_report._routing_axis_ok(None, allow_deterministic=False) is False
    assert multi_model_report._routing_axis_integrity_ok(weak_routing, allow_deterministic=False)
    assert not multi_model_report._routing_axis_integrity_ok(
        weak_routing.model_copy(update={"provenance_verified": False}), allow_deterministic=False
    )


def test_common_command_helpers_preserve_service_failure_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert common_commands.DesktopTokenBridge("secret").get_token() == "secret"
    assert common_commands._same_file(tmp_path / "missing", tmp_path / "missing") is True
    assert common_commands._effective_fake(SimpleNamespace(obj={"fake": True}), False) is True
    assert common_commands._effective_oneshot(SimpleNamespace(obj={"oneshot": True})) is True

    healthy = SimpleNamespace(ok=True)
    manager = SimpleNamespace(
        home=tmp_path,
        start=lambda **_kwargs: healthy,
        status=lambda: healthy,
        stop=lambda: healthy,
        stop_started_services=lambda _before: healthy,
    )
    monkeypatch.setattr(
        common_commands,
        "_composition",
        SimpleNamespace(
            _manager=lambda: manager,
            _print_status=lambda _status: None,
            _ensure_services=lambda _fake: healthy,
            _run_april_cli=lambda _args: 7,
        ),
    )
    assert common_commands._ensure_services(fake=True) is healthy
    with pytest.raises(typer.Exit) as exit_info:
        common_commands._delegate([], fake=True, oneshot=True)
    assert exit_info.value.exit_code == 7

    failing = SimpleNamespace(ok=False)
    monkeypatch.setattr(
        common_commands,
        "_composition",
        SimpleNamespace(
            _manager=lambda: SimpleNamespace(start=lambda **_kwargs: failing),
            _print_status=lambda _status: None,
        ),
    )
    with pytest.raises(typer.Exit) as exit_info:
        common_commands._ensure_services(fake=False)
    assert exit_info.value.exit_code == 1

    raising = SimpleNamespace(
        start=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("start failed"))
    )
    monkeypatch.setattr(
        common_commands,
        "_composition",
        SimpleNamespace(_manager=lambda: raising, _print_status=lambda _status: None),
    )
    with pytest.raises(typer.Exit) as exit_info:
        common_commands._ensure_services(fake=False)
    assert exit_info.value.exit_code == 1


def test_common_command_diagnostic_renderers_accept_realistic_payloads(capsys) -> None:
    common_commands._print_model_doctor(
        {
            "python_version": "3.11",
            "april_home_basename": "april",
            "runtime_backend": "fake",
            "llama_cpp_python_installed": False,
            "api_token": "configured",
            "runtime_token": "configured",
            "machine": "x86_64",
            "cpu_count": 4,
            "estimated_ram": "8 GB",
            "models": [
                {
                    "id": "brain",
                    "role": "brain",
                    "path": "brain.gguf",
                    "path_exists": False,
                    "file_size": 0,
                    "context_size": 1024,
                    "threads": 2,
                    "n_batch": None,
                    "keep_loaded": False,
                    "idle_unload_seconds": None,
                    "realism": "unavailable",
                }
            ],
        }
    )
    common_commands._print_model_recommendation(
        {
            "architecture": "Intel",
            "platform": "Darwin",
            "python_machine": "x86_64",
            "arm64_python": False,
            "cpu_count": 4,
            "available_memory": "8 GB",
            "recommended_profile": "cpu",
            "expected_backend": "llama_cpp",
            "notes": ["use a small model"],
            "manual_commands": ["make verify-global"],
        }
    )
    output = capsys.readouterr().out
    assert "Configured Models" in output
    assert "make verify-global" in output


def test_common_command_process_and_browser_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(common_commands.webbrowser, "open", lambda *_args, **_kwargs: True)
    assert common_commands._open_desktop_browser("http://127.0.0.1/#token") is True
    monkeypatch.setitem(sys.modules, "webview", None)
    assert common_commands._open_desktop_native("http://127.0.0.1/", "token") is False
    manager = SimpleNamespace(home=tmp_path)
    monkeypatch.setattr(
        common_commands,
        "_composition",
        SimpleNamespace(_manager=lambda: manager),
    )
    monkeypatch.setattr(common_commands, "run_terminal_process_sync", lambda *_args, **_kwargs: 3)
    assert common_commands._run_april_cli(["status"]) == 3
    assert common_commands._same_file(tmp_path / "left", tmp_path / "right") is False


def test_common_command_renderers_cover_benchmark_and_brain_failure_rows(capsys) -> None:
    common_commands._print_benchmark(
        [
            BenchmarkResult(
                run_index=1,
                load_time_seconds=1.2,
                first_token_latency_seconds=None,
                generation_time_seconds=2.3,
                output_tokens=4,
                tokens_per_second=1.7,
                unload_success=False,
                detail="degraded",
            )
        ]
    )
    common_commands._print_brain_eval(
        [
            SimpleNamespace(
                id="case-1",
                ok=False,
                expected_intent="planning",
                expected_agent="general_agent",
                actual={"intent": "unknown", "agent": "unknown"},
                detail="mismatch",
            )
        ]
    )
    output = capsys.readouterr().out
    assert "CPU-only recommendation" in output
    assert "case-1" in output
    assert "mismatch" in output


def test_launcher_doctor_reports_invalid_configuration_without_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    manager = SimpleNamespace(home=tmp_path, status=lambda: SimpleNamespace())
    monkeypatch.setattr(
        common_commands,
        "_composition",
        SimpleNamespace(_manager=lambda: manager, _print_status=lambda _status: None),
    )
    monkeypatch.setattr(
        common_commands,
        "load_settings",
        lambda **_kwargs: (_ for _ in ()).throw(ConfigError("bad")),
    )
    monkeypatch.setattr(common_commands.shutil, "which", lambda _name: None)

    common_commands._doctor()

    output = capsys.readouterr().out
    assert "APRIL Launcher Doctor" in output
    assert "run was not found in PATH" in output


def test_local_security_checks_report_each_dependency_without_live_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = SimpleNamespace(
        home=tmp_path,
        environment="development",
        workers=SimpleNamespace(development_unsandboxed_override=False),
        security=SimpleNamespace(credential_store="auto"),
        api=SimpleNamespace(token="api"),
        runtime=SimpleNamespace(token="runtime"),
        database_path=tmp_path / "april.db",
    )
    monkeypatch.setattr(
        "apps.runner.verification.local_checks.load_settings",
        lambda **_kwargs: settings,
    )
    monkeypatch.setattr(
        "apps.runner.verification.local_checks.sandbox_capabilities",
        lambda **_kwargs: SimpleNamespace(
            backend=SimpleNamespace(value="seatbelt"),
            network_denial_available=True,
            filesystem_policy_available=True,
            production_fail_closed=True,
            development_override_enabled=False,
            warning=None,
        ),
    )
    monkeypatch.setattr(
        "apps.runner.verification.local_checks.legacy_plaintext_credentials_detected",
        lambda _home: True,
    )
    monkeypatch.setattr(
        "apps.runner.verification.local_checks.audit_logger_for_settings",
        lambda *_args, **_kwargs: SimpleNamespace(
            verify=lambda: SimpleNamespace(valid=True, status="valid")
        ),
    )
    monkeypatch.setattr(
        "apps.runner.verification.local_checks.check_database",
        lambda *_args, **_kwargs: SimpleNamespace(
            last_successful_backup={"creation_timestamp": "now"},
            quick_check="ok",
            foreign_key_consistent=True,
            foreign_key_violations=0,
            journal_mode="wal",
        ),
    )
    monkeypatch.setattr(
        "apps.runner.verification.local_checks.inspect_rollout_state",
        lambda _settings: {"status": "ok"},
    )

    sandbox = run_local_sandbox_verification(tmp_path)
    security = run_local_security_integrity_verification(tmp_path)

    assert all(check.ok for check in sandbox)
    legacy_check = next(check for check in security if check.name == "legacy plaintext credential")
    assert legacy_check.ok is False
    assert any(
        check.name == "last successful backup" and check.detail == "now" for check in security
    )


def test_routing_evidence_rejects_malformed_or_oversized_error_codes() -> None:
    malformed = httpx.Response(503, request=httpx.Request("GET", "http://test"), text="offline")
    assert _downstream_error_code(malformed) is None
    non_dict = httpx.Response(
        503, request=httpx.Request("GET", "http://test"), json={"error": "bad"}
    )
    assert _downstream_error_code(non_dict) is None
    nested_bad = httpx.Response(
        503,
        request=httpx.Request("GET", "http://test"),
        json={"error": {"details": {"error": {"code": "lower-case"}}}},
    )
    assert _downstream_runtime_error_code(nested_bad) is None
    assert _routing_stage_code({}, malformed) == "missing_correlated_route_event"
    assert (
        _routing_stage_code(
            {"route_source": "fallback", "routing_failure_code": "inference_transport_error"},
            httpx.Response(
                502,
                request=httpx.Request("GET", "http://test"),
                text="runtime unavailable",
            ),
        )
        == "runtime_unavailable"
    )


def test_report_database_helpers_fail_closed_on_corrupt_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "corrupt.db"
    database.write_bytes(b"not sqlite")
    monkeypatch.setattr(
        "apps.runner.verification.reports.connect_sqlite",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.Error("bad")),
    )
    assert latest_brain_decision_marker(database) == 0
    assert brain_decision_after_marker(database, 0) == {}


def test_api_report_selection_and_safe_projection_are_bounded(tmp_path: Path, monkeypatch) -> None:
    settings = SimpleNamespace(home=tmp_path)
    root = tmp_path / "data" / "verification"
    root.mkdir(parents=True)
    (root / "multi.json").write_text(
        json.dumps(
            {
                "report_type": "multi_model",
                "generated_at": "2099-01-01T00:00:00Z",
                "summary": "pass",
                "verification_level": "all",
                "real_model_verified": True,
                "models": [
                    {
                        "model_id": "brain",
                        "role": "brain",
                        "backend": "llama_cpp",
                        "path": "/private/secret/brain.gguf",
                        "available": True,
                    }
                ],
                "skipped": [{"name": "reading", "reason": "/private/secret/missing.gguf"}],
                "threshold_failures": ["/private/secret/failure"],
            }
        ),
        encoding="utf-8",
    )
    (root / "voice.json").write_text(
        json.dumps(
            {
                "report_type": "voice_live",
                "generated_at": "2099-01-02T00:00:00Z",
                "summary": "pass",
                "evidence_mode": "real_hardware",
                "voice_live_verified": True,
                "recording_success": True,
                "stt_success": True,
                "tts_success": True,
                "playback_user_confirmed": True,
                "transcript": "secret",
            }
        ),
        encoding="utf-8",
    )
    (root / "bad.json").write_text("not-json", encoding="utf-8")
    monkeypatch.setattr("services.api.reporting.config_fingerprint_digest", lambda _home: "fp")

    latest = api_reporting._latest_verification_report(settings, report_type="real_model")
    history = api_reporting._verification_report_history(settings)
    browser = api_reporting._browser_reports(settings)
    voice_flags = api_reporting._latest_live_voice_flags(settings)
    freshness = api_reporting._reports_freshness(settings)

    assert latest["status"] == "ok"
    assert latest["report"]["basename"] == "multi.json"
    assert str(tmp_path) not in json.dumps(latest)
    assert history["count"] == 2
    assert browser["count"] == 2
    assert voice_flags["voice_live_verified"] is True
    assert freshness["multi_model"]["basename"] == "multi.json"

    projected = api_reporting._safe_report_payload(
        {
            "report_type": "workflow",
            "summary": "fail",
            "real_model_exercised": True,
            "checks": [
                {"name": "route", "status": "fail", "ok": False, "detail": "Bearer secret"},
                "ignore",
            ],
            "real_model": {"attempted": True, "path_basename": "brain.gguf"},
        },
        root / "workflow.json",
    )
    assert projected["checks"][0]["detail"] == "sensitive detail redacted"
    assert projected["file_basename"] == "workflow.json"


def test_api_report_path_and_browser_filters_fail_closed(tmp_path: Path) -> None:
    settings = SimpleNamespace(home=tmp_path)
    root = tmp_path / "data" / "verification"
    root.mkdir(parents=True)
    report_path = root / "valid.json"
    report_path.write_text(
        json.dumps(
            {
                "report_type": "go_live",
                "final_status": "warning",
                "services": {"requested": True},
            }
        ),
        encoding="utf-8",
    )
    assert api_reporting._browser_latest(settings)["status"] == "ok"
    assert api_reporting._browser_latest(settings, report_type="missing")["status"] == "not_found"
    summary = api_reporting._browser_report_summary(
        {
            "report_type": "go_live",
            "final_status": "warning",
            "services": {"requested": True, "mode": "fake", "api_reachable": True},
            "hardening_warnings": ["warning"],
        },
        report_path,
    )
    assert summary["services"]["api_reachable"] is True
    assert summary["hardened_go_live_ready"] is False
    with pytest.raises(HTTPException, match="unsafe report basename"):
        api_reporting._safe_report_path(settings, "../valid.json")
    with pytest.raises(HTTPException, match="not found"):
        api_reporting._verification_report_detail(settings, "missing.json")
