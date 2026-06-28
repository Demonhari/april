from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from apps.runner.install import install_wrappers
from apps.runner.main import app
from apps.runner.model_tools import setup_model_set, setup_voice_stack
from apps.runner.service_manager import ServiceInfo, ServiceStatus
from apps.runner.soak import SoakReport
from apps.runner.verify import BenchmarkResult, VerifyCheck
from apps.runner.voice_live import VoiceLiveReport
from apps.runner.wake_live import WakeWordLiveReport
from april_common.config_validation import validate_configuration
from april_common.errors import ConfigError
from april_common.settings import load_settings


class FakeManager:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.started: list[bool] = []
        self.stopped = False
        self.restarted: list[bool] = []

    def _status(self, *, ok: bool = True) -> ServiceStatus:
        return ServiceStatus(
            runtime=ServiceInfo(
                name="runtime",
                pid=111 if ok else None,
                running=ok,
                healthy=ok,
                log_path=self.home / "logs" / "runtime.log",
            ),
            api=ServiceInfo(
                name="api",
                pid=222 if ok else None,
                running=ok,
                healthy=ok,
                log_path=self.home / "logs" / "api.log",
            ),
        )

    def start(self, *, fake_backend: bool = False) -> ServiceStatus:
        self.started.append(fake_backend)
        return self._status()

    def status(self) -> ServiceStatus:
        return self._status(ok=False)

    def stop(self) -> ServiceStatus:
        self.stopped = True
        return self._status(ok=False)

    def restart(self, *, fake_backend: bool = False) -> ServiceStatus:
        self.restarted.append(fake_backend)
        return self._status()

    def logs(self, *, lines: int = 80) -> str:
        return f"runtime log\napi log\nlines={lines}"


def test_run_april_status_does_not_start_services(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "status"])
    assert result.exit_code == 0
    assert manager.started == []
    assert "runtime" in result.output


def test_run_april_status_json(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "status", "--json"])
    assert result.exit_code == 0
    assert '"ok"' in result.output


def test_run_april_ask_delegates_after_services(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    delegated: list[list[str]] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main._run_april_cli", lambda args: delegated.append(args) or 0)
    result = CliRunner().invoke(app, ["april", "ask", "April, plan my work today."])
    assert result.exit_code == 0
    assert manager.started == [False]
    assert delegated == [["ask", "April, plan my work today."]]


def test_run_april_oneshot_stops_started_services(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    delegated: list[list[str]] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main._run_april_cli", lambda args: delegated.append(args) or 0)
    result = CliRunner().invoke(
        app,
        ["april", "--fake", "--oneshot", "ask", "April, plan my work today."],
    )
    assert result.exit_code == 0
    assert manager.started == [True]
    assert manager.stopped is True
    assert delegated == [["ask", "April, plan my work today."]]


def test_run_april_fake_reaches_service_start(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    delegated: list[list[str]] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main._run_april_cli", lambda args: delegated.append(args) or 0)
    result = CliRunner().invoke(app, ["april", "ask", "hello", "--fake"])
    assert result.exit_code == 0
    assert manager.started == [True]
    assert delegated == [["ask", "hello"]]


def test_run_april_logs_prints_recent_logs(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "logs", "--lines", "150"])
    assert result.exit_code == 0
    assert "runtime log" in result.output
    assert "lines=150" in result.output
    result = CliRunner().invoke(app, ["april", "logs", "--tail", "100"])
    assert result.exit_code == 0
    assert "lines=100" in result.output


def test_run_april_stop_calls_manager(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "stop"])
    assert result.exit_code == 0
    assert manager.stopped is True


def test_run_april_model_load_delegates(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    delegated: list[list[str]] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main._run_april_cli", lambda args: delegated.append(args) or 0)
    result = CliRunner().invoke(app, ["april", "model", "load", "april-brain", "--fake"])
    assert result.exit_code == 0
    assert manager.started == [True]
    assert delegated == [["model", "load", "april-brain"]]


def test_run_april_project_and_memory_commands_delegate(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    delegated: list[list[str]] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main._run_april_cli", lambda args: delegated.append(args) or 0)
    runner = CliRunner()
    assert runner.invoke(app, ["april", "projects", "--fake"]).exit_code == 0
    assert runner.invoke(app, ["april", "project", "add", str(tmp_path), "--fake"]).exit_code == 0
    assert runner.invoke(app, ["april", "memory", "search", "query", "--fake"]).exit_code == 0
    assert delegated == [
        ["projects"],
        ["project", "add", str(tmp_path)],
        ["memory", "search", "query"],
    ]


def test_run_april_voice_reminder_and_task_commands_delegate(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    delegated: list[list[str]] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main._run_april_cli", lambda args: delegated.append(args) or 0)
    runner = CliRunner()
    assert runner.invoke(app, ["april", "voice", "health", "--fake"]).exit_code == 0
    assert runner.invoke(app, ["april", "voice", "doctor", "--fake"]).exit_code == 0
    assert runner.invoke(app, ["april", "voice", "ptt", "--seconds", "2", "--fake"]).exit_code == 0
    assert (
        runner.invoke(app, ["april", "voice", "test-record", "--seconds", "3", "--fake"]).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app, ["april", "voice", "test-stt", str(tmp_path / "a.wav"), "--fake"]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["april", "voice", "test-tts", "Hello Hari", "--fake"]).exit_code == 0
    assert runner.invoke(app, ["april", "reminder", "list", "--fake"]).exit_code == 0
    assert (
        runner.invoke(
            app,
            [
                "april",
                "reminder",
                "create",
                "stand up",
                "--due-at",
                "2026-06-21T09:00:00Z",
                "--fake",
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(app, ["april", "reminder", "delete", "reminder-1", "--fake"]).exit_code == 0
    )
    assert runner.invoke(app, ["april", "task", "list", "--fake"]).exit_code == 0
    assert delegated == [
        ["voice", "health"],
        ["voice", "doctor"],
        ["voice", "ptt", "--seconds", "2.0"],
        ["voice", "test-record", "--seconds", "3.0"],
        ["voice", "test-stt", str(tmp_path / "a.wav")],
        ["voice", "test-tts", "Hello Hari"],
        ["reminder", "list"],
        ["reminder", "create", "stand up", "--due-at", "2026-06-21T09:00:00Z"],
        ["reminder", "delete", "reminder-1"],
        ["task", "list"],
    ]


def test_run_april_voice_verify_live_uses_local_verifier(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    manager.settings = load_settings(root=tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.main.collect_voice_doctor",
        lambda settings: {"status": "ok", "macos_microphone_permission_guidance": "guidance"},
    )
    captured: dict[str, object] = {}

    async def _fake_voice_live(**kwargs: object) -> VoiceLiveReport:
        captured.update(kwargs)
        return VoiceLiveReport(
            timestamp="2026-06-26T00:00:00Z",
            platform="Darwin 24",
            sounddevice_available=True,
            input_device_count=1,
            output_device_count=1,
            whisper_binary_available=True,
            whisper_model_available=True,
            piper_binary_available=True,
            piper_model_available=True,
            wake_word_model_available=False,
            recording_success=True,
            stt_success=True,
            transcript_length=10,
            transcription_user_confirmed=True,
            tts_success=True,
            playback_user_confirmed=True,
            summary="pass",
        )

    monkeypatch.setattr("apps.runner.main.run_voice_live_verification", _fake_voice_live)
    out = tmp_path / "voice-live.json"
    result = CliRunner().invoke(
        app,
        ["april", "voice", "verify-live", "--seconds", "1", "--report", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert captured["settings"] is manager.settings
    assert captured["seconds"] == 1
    assert captured["report_path"] == out
    assert "transcript_length=10" in result.output


def test_run_april_voice_verify_wake_live_uses_local_verifier(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    manager.settings = load_settings(root=tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.main.collect_voice_doctor",
        lambda settings: {
            "status": "ok",
            "macos_microphone_permission_guidance": "guidance",
            "wake_word_guidance": "say April",
        },
    )
    captured: dict[str, object] = {}

    async def _fake_wake_live(**kwargs: object) -> WakeWordLiveReport:
        captured.update(kwargs)
        return WakeWordLiveReport(
            summary="pass",
            wake_word_configured=True,
            wake_word_detected=True,
            recording_success=True,
            stt_success=True,
            transcript_length=12,
            normalized_transcript_length=8,
            api_success=True,
            tts_success=True,
            playback_user_confirmed=True,
            wake_word_live_verified=True,
        )

    monkeypatch.setattr("apps.runner.main.run_wake_word_live_verification", _fake_wake_live)
    out = tmp_path / "wake-live.json"
    result = CliRunner().invoke(
        app,
        ["april", "voice", "verify-wake-live", "--wake-wait-seconds", "5", "--report", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert captured["settings"] is manager.settings
    assert captured["wake_wait_seconds"] == 5
    assert captured["report_path"] == out
    assert "wake_word_detected=True" in result.output


def test_run_april_voice_verify_wake_live_exits_nonzero_on_fail(
    tmp_path: Path, monkeypatch
) -> None:
    manager = FakeManager(tmp_path)
    manager.settings = load_settings(root=tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.main.collect_voice_doctor", lambda settings: {"status": "degraded"}
    )

    async def _fake_wake_live(**kwargs: object) -> WakeWordLiveReport:
        return WakeWordLiveReport(summary="fail")

    monkeypatch.setattr("apps.runner.main.run_wake_word_live_verification", _fake_wake_live)
    result = CliRunner().invoke(app, ["april", "voice", "verify-wake-live"])
    assert result.exit_code == 1


def test_run_april_config_validate_reports_success(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main.validate_configuration", lambda home: [])
    result = CliRunner().invoke(app, ["april", "config", "validate"])
    assert result.exit_code == 0
    assert "configuration is valid" in result.output


def test_run_april_config_validate_reports_errors(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main.validate_configuration", lambda home: ["bad config"])
    result = CliRunner().invoke(app, ["april", "config", "validate"])
    assert result.exit_code == 1
    assert "bad config" in result.output


def test_run_april_config_inspect_redacts_token(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main.validate_configuration", lambda home: [])
    monkeypatch.setattr(
        "apps.runner.main.ModelRegistry.from_file",
        lambda path, *, root: type("FakeModels", (), {"list": lambda self: []})(),
    )
    result = CliRunner().invoke(app, ["april", "config", "inspect"])
    assert result.exit_code == 0
    assert "[REDACTED]" in result.output
    assert "local-dev-token" not in result.output
    assert "local-dev-runtime-token" not in result.output


def test_run_april_verify_fake_reports_table(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.main.run_fake_verification",
        lambda home: [VerifyCheck(name="runtime health", ok=True, detail="ok")],
    )
    result = CliRunner().invoke(app, ["april", "verify", "--fake"])
    assert result.exit_code == 0
    assert "APRIL Verification" in result.output
    assert "runtime health" in result.output


def test_run_april_verify_fake_soak_short_mode(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    captured: dict[str, object] = {}

    def _fake_soak(home: Path, **kwargs: object) -> SoakReport:
        captured["home"] = home
        captured.update(kwargs)
        return SoakReport(
            generated_at="2026-06-26T00:00:00Z",
            duration_seconds=0.6,
            iterations=2,
            latency_ms={"median": 1.0},
            summary="pass",
        )

    monkeypatch.setattr("apps.runner.main.run_fake_soak", _fake_soak)
    out = tmp_path / "soak.json"
    result = CliRunner().invoke(
        app,
        [
            "april",
            "verify",
            "--soak",
            "--fake",
            "--minutes",
            "0.01",
            "--report",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["home"] == tmp_path
    assert captured["minutes"] == 0.01
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["report_type"] == "soak"
    assert data["real_model_verified"] is False
    assert "APRIL Fake Soak Verification" in result.output


def test_run_april_verify_target_mac_writes_report(tmp_path: Path, monkeypatch) -> None:
    from apps.runner.mac_report import MacVerificationReport, RealModelReport

    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    captured: dict[str, object] = {}

    class _StubValidator:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

        def run(self) -> list[VerifyCheck]:
            return [VerifyCheck(name="machine architecture", ok=True, detail="Darwin/arm64")]

        def build_report(self, *, thresholds: object) -> MacVerificationReport:
            captured["thresholds"] = thresholds
            return MacVerificationReport(
                generated_at="t",
                os="Darwin 24",
                cpu_architecture="arm64",
                python_version="3.11.15",
                runtime_backend="llama_cpp",
                real_model=RealModelReport(attempted=False),
                summary="degraded",
            )

    monkeypatch.setattr("apps.runner.main.TargetMacValidator", _StubValidator)
    out = tmp_path / "verification" / "report.json"
    result = CliRunner().invoke(
        app,
        ["april", "verify", "--target-mac", "--report", str(out), "--min-tokens-per-second", "5"],
    )
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"] == "degraded"
    assert data["real_model"]["attempted"] is False
    assert "summary: degraded" in result.output
    assert captured["thresholds"].min_tokens_per_second == 5  # type: ignore[attr-defined]


def test_run_april_verify_all_configured_cli_flags(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    captured: dict[str, object] = {}

    class _StubAllVerifier:
        def __init__(self) -> None:
            self.checks = [
                VerifyCheck(name="model april-brain acceptance gates", ok=True, detail="ok")
            ]

        def build_report(self) -> object:
            raise AssertionError("report should not be built without --report")

    def _fake_all_configured(home: Path, **kwargs: object) -> _StubAllVerifier:
        captured["home"] = home
        captured.update(kwargs)
        return _StubAllVerifier()

    monkeypatch.setattr(
        "apps.runner.main.run_all_configured_models_verification",
        _fake_all_configured,
    )
    result = CliRunner().invoke(
        app,
        [
            "april",
            "verify",
            "--all-configured-models",
            "--require-real-model",
            "--max-rss-mb",
            "4096",
            "--min-routing-accuracy",
            "0.95",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["home"] == tmp_path
    assert captured["require_real_model"] is True
    thresholds = captured["thresholds"]
    assert thresholds.max_rss_mb == 4096  # type: ignore[attr-defined]
    assert thresholds.min_routing_accuracy == 0.95  # type: ignore[attr-defined]
    assert "APRIL All-Configured-Model Verification" in result.output


def test_run_april_verify_mac_readiness_alias(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    called: list[bool] = []

    class _StubAllVerifier:
        def __init__(self) -> None:
            self.checks = [VerifyCheck(name="configured models", ok=True, detail="ok")]

        def build_report(self) -> object:
            raise AssertionError("report should not be built without --report")

    monkeypatch.setattr(
        "apps.runner.main.run_all_configured_models_verification",
        lambda home, **kwargs: called.append(True) or _StubAllVerifier(),
    )
    result = CliRunner().invoke(app, ["april", "verify", "--mac-readiness"])
    assert result.exit_code == 0, result.output
    assert called == [True]


def test_run_april_verify_fake_fails_on_failed_check(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.main.run_fake_verification",
        lambda home: [VerifyCheck(name="runtime health", ok=False, detail="offline")],
    )
    result = CliRunner().invoke(app, ["april", "verify", "--fake"])
    assert result.exit_code == 1
    assert "offline" in result.output


def test_run_april_verify_workflow_json_and_failure(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    captured: dict[str, object] = {}

    def _fake_workflow(home: Path, **kwargs: object) -> list[VerifyCheck]:
        captured.update(kwargs)
        return [VerifyCheck(name="workflow", ok=True, detail=str(kwargs))]

    monkeypatch.setattr(
        "apps.runner.main.run_workflow_verification",
        _fake_workflow,
    )
    result = CliRunner().invoke(
        app,
        [
            "april",
            "verify",
            "--workflow",
            "--real-model",
            str(tmp_path / "model.gguf"),
            "--max-output-tokens",
            "77",
            "--timeout",
            "9.5",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert "workflow" in result.output
    assert captured["real_model"] is True
    assert captured["max_output_tokens"] == 77
    assert captured["timeout"] == 9.5
    monkeypatch.setattr(
        "apps.runner.main.run_workflow_verification",
        lambda home, **kwargs: [VerifyCheck(name="workflow", ok=False, detail="bad")],
    )
    failed = CliRunner().invoke(app, ["april", "verify", "--workflow"])
    assert failed.exit_code == 1
    assert "bad" in failed.output


def test_run_april_verify_workflow_writes_report(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.main.run_workflow_verification",
        lambda home, **kwargs: [
            VerifyCheck(name="real workflow planning route", ok=True, detail="/tmp/private/path"),
            VerifyCheck(name="workflow voice health", ok=True, detail="disabled"),
        ],
    )
    out = tmp_path / "workflow.json"
    result = CliRunner().invoke(
        app,
        [
            "april",
            "verify",
            "--workflow",
            "--real-model",
            str(tmp_path / "m.gguf"),
            "--report",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["report_type"] == "workflow"
    assert payload["real_model_verified"] is True
    assert payload["checks_failed"] == 0
    assert str(tmp_path) not in json.dumps(payload)


def test_run_april_verify_target_mac_json(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)

    class _StubValidator:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self) -> list[VerifyCheck]:
            return [
                VerifyCheck(name="machine architecture", ok=True, detail="Darwin/arm64"),
                VerifyCheck(name="voice smoke", ok=True, detail="manual", status="manual"),
            ]

    monkeypatch.setattr("apps.runner.main.TargetMacValidator", _StubValidator)
    result = CliRunner().invoke(app, ["april", "verify", "--target-mac", "--json"])
    assert result.exit_code == 0
    assert '"status": "manual"' in result.output
    assert "machine architecture" in result.output


def test_run_april_verify_real_model_skips_without_path(monkeypatch) -> None:
    monkeypatch.delenv("APRIL_TEST_GGUF_PATH", raising=False)
    result = CliRunner().invoke(app, ["april", "verify", "--real-model"])
    assert result.exit_code == 0
    assert "Skipping real-model verification" in result.output


def test_run_april_verify_real_model_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.gguf"
    result = CliRunner().invoke(app, ["april", "verify", "--real-model", str(missing)])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_run_april_verify_real_model_existing_path_runs_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"not a real model")
    manager = FakeManager(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.main.run_real_model_verification",
        lambda home, path, **_kwargs: (
            calls.append(path) or [VerifyCheck(name="real model chat", ok=True, detail="ok")]
        ),
    )
    result = CliRunner().invoke(app, ["april", "verify", "--real-model", str(gguf)])
    assert result.exit_code == 0
    assert calls == [gguf]
    assert "APRIL Real Model Verification" in result.output


def _copy_configs(home: Path) -> None:
    shutil.copytree(Path.cwd() / "configs", home / "configs")


def test_readiness_cli_human_and_json_are_offline_and_redacted(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    runner = CliRunner()
    human = runner.invoke(app, ["april", "readiness"])
    assert human.exit_code == 0, human.output
    assert "readiness" in human.output.lower()
    assert "run april verify" in human.output

    structured = runner.invoke(app, ["april", "readiness", "--json"])
    assert structured.exit_code == 0, structured.output
    payload = json.loads(structured.output)
    assert "real_model_ready" in payload
    assert payload["real_model_ready"] is False
    assert payload["voice_ready"] is False
    assert "real_model_preflight_ready" in payload
    assert "voice_preflight_ready" in payload
    assert "blockers" in payload
    assert "next_actions" in payload
    # Offline + redacted: no absolute home path, no token value in JSON output.
    assert str(tmp_path) not in structured.output
    assert "local-dev-token" not in structured.output


def test_setup_models_dry_run_writes_nothing_and_prints_basenames(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    model_dir = tmp_path / "secret-models"
    model_dir.mkdir()
    brain = model_dir / "brain.gguf"
    brain.write_bytes(b"gguf")
    before = (tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "setup", "models", "--brain", str(brain)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8") == before
    assert "dry run" in result.output
    assert "brain.gguf" in result.output
    assert "secret-models" not in result.output
    assert "run april model doctor" in result.output


def test_setup_models_apply_writes_config_and_backup(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    brain = tmp_path / "brain.gguf"
    brain.write_bytes(b"gguf")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app,
        ["april", "setup", "models", "--brain", str(brain), "--apply"],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load((tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert data["models"]["brain"]["id"] == "april-brain"
    assert data["models"]["brain"]["path"] == str(brain.resolve())
    assert list((tmp_path / "configs").glob("models.yaml.bak-*"))
    assert validate_configuration(tmp_path) == []


def test_setup_models_rejects_missing_and_non_gguf(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    missing = CliRunner().invoke(
        app,
        ["april", "setup", "models", "--brain", str(tmp_path / "missing.gguf")],
    )
    assert missing.exit_code == 1
    assert "missing.gguf" in missing.output
    text = tmp_path / "model.txt"
    text.write_text("bad", encoding="utf-8")
    non_gguf = CliRunner().invoke(app, ["april", "setup", "models", "--brain", str(text)])
    assert non_gguf.exit_code == 1
    assert ".gguf" in non_gguf.output


def test_setup_models_rejects_symlink_escape(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"gguf")
    link = home / "linked.gguf"
    link.symlink_to(outside)
    manager = FakeManager(home)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "setup", "models", "--brain", str(link)])
    assert result.exit_code == 1
    assert "symlink target is outside" in result.output


def test_setup_models_copy_into_models_uses_local_home(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    source = tmp_path / "source.gguf"
    source.write_bytes(b"gguf")
    manager = FakeManager(home)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app,
        [
            "april",
            "setup",
            "models",
            "--brain",
            str(source),
            "--copy-into-models",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (home / "models" / "source.gguf").exists()
    data = yaml.safe_load((home / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert data["models"]["brain"]["path"] == "models/source.gguf"


def test_setup_model_set_copy_rollback_removes_new_copy_and_restores_config(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    brain = source_dir / "brain.gguf"
    coding = source_dir / "coding.gguf"
    brain.write_bytes(b"brain")
    coding.write_bytes(b"coding")
    models_dir = home / "models"
    models_dir.mkdir()
    existing_coding = models_dir / "coding.gguf"
    existing_coding.write_bytes(b"existing")
    before = (home / "configs" / "models.yaml").read_bytes()

    with pytest.raises(ConfigError):
        setup_model_set(
            home=home,
            role_paths={"brain": brain, "coding": coding},
            copy_into_models=True,
            apply=True,
        )

    assert not (models_dir / "brain.gguf").exists()
    assert existing_coding.read_bytes() == b"existing"
    assert (home / "configs" / "models.yaml").read_bytes() == before
    assert brain.exists()
    assert coding.exists()


def test_setup_model_set_copy_rollback_does_not_delete_preexisting_destination(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    models_dir = home / "models"
    models_dir.mkdir()
    existing_brain = models_dir / "brain.gguf"
    existing_brain.write_bytes(b"existing brain")
    existing_coding = models_dir / "coding.gguf"
    existing_coding.write_bytes(b"existing coding")
    coding = source_dir / "coding.gguf"
    coding.write_bytes(b"new coding")
    before = (home / "configs" / "models.yaml").read_bytes()

    with pytest.raises(ConfigError):
        setup_model_set(
            home=home,
            role_paths={"brain": existing_brain, "coding": coding},
            copy_into_models=True,
            apply=True,
        )

    assert existing_brain.read_bytes() == b"existing brain"
    assert existing_coding.read_bytes() == b"existing coding"
    assert (home / "configs" / "models.yaml").read_bytes() == before


def test_setup_model_set_copy_dry_run_copies_nothing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    source = tmp_path / "source.gguf"
    source.write_bytes(b"gguf")
    before = (home / "configs" / "models.yaml").read_bytes()

    result = setup_model_set(
        home=home,
        role_paths={"brain": source},
        copy_into_models=True,
        apply=False,
    )

    assert result["applied"] is False
    assert not (home / "models" / "source.gguf").exists()
    assert (home / "configs" / "models.yaml").read_bytes() == before


def test_setup_model_set_successful_copy_keeps_file_and_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    source = tmp_path / "source.gguf"
    source.write_bytes(b"gguf")

    result = setup_model_set(
        home=home,
        role_paths={"brain": source},
        copy_into_models=True,
        apply=True,
    )

    copied = home / "models" / "source.gguf"
    assert result["applied"] is True
    assert copied.exists()
    data = yaml.safe_load((home / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert data["models"]["brain"]["path"] == "models/source.gguf"


def test_setup_voice_dry_run_apply_and_missing_wake_word(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    whisper_bin = tmp_path / "whisper-main"
    whisper_model = tmp_path / "ggml-base.en.bin"
    piper_bin = tmp_path / "piper"
    piper_model = tmp_path / "voice.onnx"
    for path in (whisper_bin, whisper_model, piper_bin, piper_model):
        path.write_bytes(b"asset")
    piper_model.with_name(f"{piper_model.name}.json").write_bytes(b"{}")
    before = (tmp_path / "configs" / "april.yaml").read_text(encoding="utf-8")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    runner = CliRunner()
    base_args = [
        "april",
        "setup",
        "voice",
        "--whisper-binary",
        str(whisper_bin),
        "--whisper-model",
        str(whisper_model),
        "--piper-binary",
        str(piper_bin),
        "--piper-model",
        str(piper_model),
        "--wake-word-model",
        str(tmp_path / "missing-wake.onnx"),
    ]
    dry = runner.invoke(app, base_args)
    assert dry.exit_code == 0, dry.output
    assert (tmp_path / "configs" / "april.yaml").read_text(encoding="utf-8") == before
    assert "missing-wake.onnx" not in dry.output
    assert "wake-word model missing" in dry.output
    assert "local-dev-token" not in dry.output
    applied = runner.invoke(app, [*base_args, "--apply"])
    assert applied.exit_code == 0, applied.output
    data = yaml.safe_load((tmp_path / "configs" / "april.yaml").read_text(encoding="utf-8"))
    assert data["voice"]["whisper_binary_path"] == str(whisper_bin.resolve())
    assert data["voice"]["piper_model_path"] == str(piper_model.resolve())
    assert data["voice"]["wake_word_model_path"] is None
    assert list((tmp_path / "configs").glob("april.yaml.bak-*"))


def test_setup_voice_missing_required_path_fails(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    before = config.read_text(encoding="utf-8")
    whisper_model = tmp_path / "ggml-base.en.bin"
    piper_bin = tmp_path / "piper"
    piper_model = tmp_path / "voice.onnx"
    for path in (whisper_model, piper_bin, piper_model):
        path.write_bytes(b"asset")
    piper_model.with_name(f"{piper_model.name}.json").write_bytes(b"{}")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app,
        [
            "april",
            "setup",
            "voice",
            "--whisper-binary",
            str(tmp_path / "missing-whisper"),
            "--whisper-model",
            str(whisper_model),
            "--piper-binary",
            str(piper_bin),
            "--piper-model",
            str(piper_model),
        ],
    )
    assert result.exit_code == 1
    assert "missing-whisper" in result.output
    assert config.read_text(encoding="utf-8") == before


def test_setup_voice_missing_piper_companion_config_fails(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    before = config.read_bytes()
    whisper_bin = tmp_path / "whisper-main"
    whisper_model = tmp_path / "ggml-base.en.bin"
    piper_bin = tmp_path / "piper"
    piper_model = tmp_path / "voice.onnx"
    for path in (whisper_bin, whisper_model, piper_bin, piper_model):
        path.write_bytes(b"asset")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)

    result = CliRunner().invoke(
        app,
        [
            "april",
            "setup",
            "voice",
            "--whisper-binary",
            str(whisper_bin),
            "--whisper-model",
            str(whisper_model),
            "--piper-binary",
            str(piper_bin),
            "--piper-model",
            str(piper_model),
        ],
    )

    assert result.exit_code == 1
    assert "voice.onnx.json" in result.output
    assert config.read_bytes() == before


def test_setup_voice_stack_apply_missing_required_path_preserves_config_bytes(
    tmp_path: Path,
) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    before = config.read_bytes()
    whisper_model = tmp_path / "ggml-base.en.bin"
    piper_bin = tmp_path / "piper"
    piper_model = tmp_path / "voice.onnx"
    for path in (whisper_model, piper_bin, piper_model):
        path.write_bytes(b"asset")
    piper_model.with_name(f"{piper_model.name}.json").write_bytes(b"{}")

    with pytest.raises(ConfigError, match="missing-whisper"):
        setup_voice_stack(
            home=tmp_path,
            whisper_binary=tmp_path / "missing-whisper",
            whisper_model=whisper_model,
            piper_binary=piper_bin,
            piper_model=piper_model,
            apply=True,
        )

    assert config.read_bytes() == before


def test_setup_voice_stack_apply_missing_required_path_preserves_existing_enabled_true(
    tmp_path: Path,
) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["voice"]["enabled"] = True
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    before = config.read_bytes()
    whisper_model = tmp_path / "ggml-base.en.bin"
    piper_bin = tmp_path / "piper"
    piper_model = tmp_path / "voice.onnx"
    for path in (whisper_model, piper_bin, piper_model):
        path.write_bytes(b"asset")
    piper_model.with_name(f"{piper_model.name}.json").write_bytes(b"{}")

    with pytest.raises(ConfigError, match="missing-whisper"):
        setup_voice_stack(
            home=tmp_path,
            whisper_binary=tmp_path / "missing-whisper",
            whisper_model=whisper_model,
            piper_binary=piper_bin,
            piper_model=piper_model,
            apply=True,
        )

    assert config.read_bytes() == before
    assert yaml.safe_load(config.read_text(encoding="utf-8"))["voice"]["enabled"] is True


def test_setup_voice_stack_restores_config_when_post_write_validation_fails(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    before = config.read_bytes()
    args = _voice_setup_paths(tmp_path)
    monkeypatch.setattr(
        "apps.runner.model_tools.validate_configuration",
        lambda home: ["invalid voice config"],
    )

    with pytest.raises(ConfigError, match="Configuration validation failed"):
        setup_voice_stack(home=tmp_path, apply=True, enable=True, **args)

    assert config.read_bytes() == before


def _voice_setup_args(tmp_path: Path, *, wake_word: Path | None = None) -> list[str]:
    whisper_bin = tmp_path / "whisper-main"
    whisper_model = tmp_path / "ggml-base.en.bin"
    piper_bin = tmp_path / "piper"
    piper_model = tmp_path / "voice.onnx"
    for path in (whisper_bin, whisper_model, piper_bin, piper_model):
        if not path.exists():
            path.write_bytes(b"asset")
    piper_model.with_name(f"{piper_model.name}.json").write_bytes(b"{}")
    args = [
        "april",
        "setup",
        "voice",
        "--whisper-binary",
        str(whisper_bin),
        "--whisper-model",
        str(whisper_model),
        "--piper-binary",
        str(piper_bin),
        "--piper-model",
        str(piper_model),
    ]
    if wake_word is not None:
        args.extend(["--wake-word-model", str(wake_word)])
    return args


def _voice_setup_paths(tmp_path: Path) -> dict[str, Path]:
    whisper_bin = tmp_path / "whisper-main"
    whisper_model = tmp_path / "ggml-base.en.bin"
    piper_bin = tmp_path / "piper"
    piper_model = tmp_path / "voice.onnx"
    for path in (whisper_bin, whisper_model, piper_bin, piper_model):
        if not path.exists():
            path.write_bytes(b"asset")
    piper_model.with_name(f"{piper_model.name}.json").write_bytes(b"{}")
    return {
        "whisper_binary": whisper_bin,
        "whisper_model": whisper_model,
        "piper_binary": piper_bin,
        "piper_model": piper_model,
    }


def test_setup_voice_apply_without_enable_keeps_voice_disabled(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, [*_voice_setup_args(tmp_path), "--apply"])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load((tmp_path / "configs" / "april.yaml").read_text(encoding="utf-8"))
    # Paths are written, but voice stays OFF: no surprise enablement.
    assert data["voice"]["whisper_binary_path"] == str((tmp_path / "whisper-main").resolve())
    assert data["voice"]["enabled"] is False
    assert "remains DISABLED" in result.output


def test_setup_voice_apply_without_enable_disables_existing_true(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["voice"]["enabled"] = True
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, [*_voice_setup_args(tmp_path), "--apply"])
    assert result.exit_code == 0, result.output
    written = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert written["voice"]["enabled"] is False
    assert "remains DISABLED" in result.output


def test_setup_voice_dry_run_enable_leaves_existing_true_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["voice"]["enabled"] = True
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    before = config.read_text(encoding="utf-8")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, [*_voice_setup_args(tmp_path), "--dry-run", "--enable"])
    assert result.exit_code == 0, result.output
    assert config.read_text(encoding="utf-8") == before
    assert "--apply --enable" in result.output


def test_setup_voice_enable_turns_voice_on_after_validation(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    wake = tmp_path / "april.onnx"
    wake.write_bytes(b"wake")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app, [*_voice_setup_args(tmp_path, wake_word=wake), "--apply", "--enable"]
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load((tmp_path / "configs" / "april.yaml").read_text(encoding="utf-8"))
    assert data["voice"]["enabled"] is True
    assert data["voice"]["wake_word_model_path"] == str(wake.resolve())
    assert "ENABLED" in result.output
    # Even with a wake-word model present, wake-word listening stays unverified.
    assert "UNVERIFIED" in result.output


def test_setup_voice_apply_enable_missing_required_path_preserves_config(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["voice"]["enabled"] = True
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    before = config.read_bytes()
    args = _voice_setup_args(tmp_path)
    missing_index = args.index("--whisper-binary") + 1
    args[missing_index] = str(tmp_path / "missing-whisper")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, [*args, "--apply", "--enable"])
    assert result.exit_code == 1
    assert config.read_bytes() == before
    written = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert written["voice"]["enabled"] is True


def test_setup_voice_enable_without_wake_word_keeps_ptt_and_marks_wake_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    args = _voice_setup_args(tmp_path, wake_word=tmp_path / "missing-wake.onnx")
    result = CliRunner().invoke(app, [*args, "--apply", "--enable"])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load((tmp_path / "configs" / "april.yaml").read_text(encoding="utf-8"))
    assert data["voice"]["enabled"] is True
    assert data["voice"]["wake_word_model_path"] is None
    # Push-to-talk is available; wake-word listening is unavailable, not a blocker.
    assert "Push-to-talk is available" in result.output
    assert "UNAVAILABLE" in result.output
    assert "UNVERIFIED" not in result.output
    assert "wake-word model missing" in result.output


def test_setup_app_stub_command_refuses_overwrite_and_force_replaces(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    output = tmp_path / "dist" / "APRIL.app"
    runner = CliRunner()
    created = runner.invoke(app, ["april", "setup", "app-stub", "--output", str(output)])
    assert created.exit_code == 0, created.output
    launcher = output / "Contents" / "MacOS" / "APRIL"
    assert launcher.exists()
    combined = launcher.read_text(encoding="utf-8") + (
        output / "Contents" / "Info.plist"
    ).read_text(encoding="utf-8")
    for forbidden in ("local-dev-token", ".gguf", "sudo", "codesign", "notarytool"):
        assert forbidden not in combined
    again = runner.invoke(app, ["april", "setup", "app-stub", "--output", str(output)])
    assert again.exit_code == 1
    assert "--force" in again.output
    forced = runner.invoke(
        app,
        ["april", "setup", "app-stub", "--output", str(output), "--force"],
    )
    assert forced.exit_code == 0, forced.output


def test_model_import_rejects_missing_path(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app,
        [
            "april",
            "model",
            "import",
            "--role",
            "brain",
            "--id",
            "april-brain",
            "--name",
            "missing",
            "--path",
            str(tmp_path / "missing.gguf"),
        ],
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_model_import_rejects_non_gguf(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    model = tmp_path / "model.txt"
    model.write_text("not gguf", encoding="utf-8")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app,
        [
            "april",
            "model",
            "import",
            "--role",
            "brain",
            "--id",
            "april-brain",
            "--name",
            "bad",
            "--path",
            str(model),
        ],
    )
    assert result.exit_code == 1
    assert ".gguf" in result.output


def test_model_import_absolute_path_updates_config(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    model = tmp_path / "local.gguf"
    model.write_bytes(b"gguf")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(
        app,
        [
            "april",
            "model",
            "import",
            "--role",
            "brain",
            "--id",
            "april-brain",
            "--name",
            "local",
            "--path",
            str(model),
        ],
    )
    assert result.exit_code == 0
    text = (tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8")
    assert str(model) in text
    assert "run april verify --real-model" in result.output
    assert "local.gguf" in result.output
    assert validate_configuration(tmp_path) == []


def test_model_import_copy_into_models_and_no_overwrite(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    source = tmp_path / "outside" / "model.gguf"
    source.parent.mkdir()
    source.write_bytes(b"gguf")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "april",
            "model",
            "import",
            "--role",
            "brain",
            "--id",
            "april-brain",
            "--name",
            "local",
            "--path",
            str(source),
            "--copy-into-models",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "models" / "model.gguf").exists()
    again = runner.invoke(
        app,
        [
            "april",
            "model",
            "import",
            "--role",
            "brain",
            "--id",
            "april-brain",
            "--name",
            "local",
            "--path",
            str(source),
            "--copy-into-models",
        ],
    )
    assert again.exit_code == 1
    assert "--force" in again.output


def test_model_profile_list_and_apply(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    runner = CliRunner()
    listed = runner.invoke(app, ["april", "model", "profile", "list"])
    assert listed.exit_code == 0
    assert "intel_macbook_cpu_low" in listed.output
    applied = runner.invoke(app, ["april", "model", "profile", "apply", "intel_macbook_cpu_low"])
    assert applied.exit_code == 0
    text = (tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8")
    assert "context_size: 4096" in text
    assert list((tmp_path / "configs").glob("models.yaml.bak-*"))


def test_model_doctor_json(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "model", "doctor", "--json"])
    assert result.exit_code == 0
    assert "llama_cpp_python_installed" in result.output
    table = CliRunner().invoke(app, ["april", "model", "doctor"])
    assert table.exit_code == 0
    assert "APRIL Model Doctor" in table.output
    assert "Configured Models" in table.output


def test_model_benchmark_json_and_failure(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(
        "apps.runner.main.run_model_benchmark",
        lambda *args, **kwargs: [
            BenchmarkResult(
                run_index=1,
                ok=True,
                load_time_seconds=1.0,
                output_tokens=2,
                tokens_per_second=3.0,
                unload_success=True,
            )
        ],
    )
    result = CliRunner().invoke(app, ["april", "model", "benchmark", str(gguf), "--json"])
    assert result.exit_code == 0
    assert '"tokens_per_second"' in result.output
    monkeypatch.setattr(
        "apps.runner.main.run_model_benchmark",
        lambda *args, **kwargs: [BenchmarkResult(run_index=1, ok=False, detail="missing")],
    )
    failed = CliRunner().invoke(app, ["april", "model", "benchmark", str(gguf)])
    assert failed.exit_code == 1
    assert "missing" in failed.output


def test_model_benchmark_rejects_missing_path(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "model", "benchmark", str(tmp_path / "no.gguf")])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_memory_doctor(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "memory", "doctor"])
    assert result.exit_code == 0
    assert "hashed-token" in result.output
    structured = CliRunner().invoke(app, ["april", "memory", "doctor", "--json"])
    assert structured.exit_code == 0
    payload = json.loads(structured.output)
    assert payload["status"] == "ok"
    assert payload["configured_embedding_provider"] == "hashed-token"
    assert payload["active_vector_index_provider"] == "hashed-token"
    assert payload["dimensions"] == 256


def test_memory_doctor_runtime_local_without_embedding_model_not_ready(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["memory"]["embedding_provider"] = "runtime-local"
    data["memory"]["embedding_model_id"] = "april-embedding"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)

    result = CliRunner().invoke(app, ["april", "memory", "doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "not_ready"
    assert payload["runtime_local_requested"] is True
    assert payload["fell_back_to_hashed_token"] is True
    assert payload["fallback_risk"] is True
    assert payload["embedding_role_model_registered"] is False


def test_memory_doctor_mismatched_vector_provider_requires_reindex(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["memory"]["embedding_provider"] = "hashed-token"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    index = tmp_path / "data" / "vector_index"
    index.mkdir(parents=True)
    (index / "metadata.json").write_text(
        json.dumps({"provider": "runtime-local", "dimensions": 8, "record_count": 1}),
        encoding="utf-8",
    )
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)

    result = CliRunner().invoke(app, ["april", "memory", "doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["reindex_required"] is True
    assert payload["status"] == "reindex_required"
    assert payload["vector_index"]["persisted_provider"] == "runtime-local"


def test_memory_doctor_json_redacts_paths(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    model_path = tmp_path / "models" / "embed.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"GGUF")
    models = yaml.safe_load((tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8"))
    models["models"]["embedding"] = {
        "id": "april-embedding",
        "name": "embed",
        "path": "models/embed.gguf",
        "backend": "llama_cpp",
        "role": "embedding",
        "threads": 1,
        "context_size": 512,
        "temperature": 0.0,
        "max_output_tokens": 1,
        "keep_loaded": True,
    }
    (tmp_path / "configs" / "models.yaml").write_text(yaml.safe_dump(models), encoding="utf-8")
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["memory"]["embedding_provider"] = "runtime-local"
    data["memory"]["embedding_model_id"] = "april-embedding"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)

    result = CliRunner().invoke(app, ["april", "memory", "doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    blob = json.dumps(payload)
    assert payload["embedding_model_path_basename"] == "embed.gguf"
    assert str(tmp_path) not in blob


def test_memory_doctor_verify_runtime_embedding_is_explicit_and_mocked(
    tmp_path: Path, monkeypatch
) -> None:
    _copy_configs(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["memory"]["embedding_provider"] = "runtime-local"
    data["memory"]["embedding_model_id"] = "april-embedding"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    models = yaml.safe_load((tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8"))
    embed = tmp_path / "models" / "embed.gguf"
    embed.parent.mkdir()
    embed.write_bytes(b"GGUF")
    models["models"]["embedding"] = {
        "id": "april-embedding",
        "name": "embed",
        "path": "models/embed.gguf",
        "backend": "llama_cpp",
        "role": "embedding",
        "threads": 1,
        "context_size": 512,
        "temperature": 0.0,
        "max_output_tokens": 1,
        "keep_loaded": True,
    }
    (tmp_path / "configs" / "models.yaml").write_text(yaml.safe_dump(models), encoding="utf-8")
    calls: list[str | None] = []

    def _verify(settings: object, model_id: str | None) -> dict[str, object]:
        calls.append(model_id)
        return {"status": "ok", "model_id": model_id, "dimensions": 8}

    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main._verify_runtime_embedding", _verify)

    default = CliRunner().invoke(app, ["april", "memory", "doctor", "--json"])
    verified = CliRunner().invoke(
        app, ["april", "memory", "doctor", "--json", "--verify-runtime-embedding"]
    )

    assert default.exit_code == 0
    assert calls == ["april-embedding"]
    payload = json.loads(verified.output)
    assert verified.exit_code == 0
    assert payload["runtime_embedding_verification"]["status"] == "ok"
    assert payload["dimensions"] == 8


def test_eval_brain_fake_json_and_errors(tmp_path: Path, monkeypatch) -> None:
    _copy_configs(tmp_path)
    fixtures = tmp_path / "tests" / "fixtures" / "evals"
    fixtures.mkdir(parents=True)
    shutil.copy2(Path.cwd() / "tests" / "fixtures" / "evals" / "brain_routes.yaml", fixtures)
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "eval", "brain", "--fake", "--json"])
    assert result.exit_code == 0
    assert "normal_chat" in result.output
    table = CliRunner().invoke(app, ["april", "eval", "brain", "--fake"])
    assert table.exit_code == 0
    assert "APRIL Brain Eval" in table.output
    missing_mode = CliRunner().invoke(app, ["april", "eval", "brain"])
    assert missing_mode.exit_code == 1
    missing_model = CliRunner().invoke(
        app,
        ["april", "eval", "brain", "--real-model", str(tmp_path / "missing.gguf")],
    )
    assert missing_model.exit_code == 1


def test_doctor_reports_missing_path(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    empty_bin = tmp_path / "empty-bin"
    home.mkdir()
    repo.mkdir()
    empty_bin.mkdir()
    install_wrappers(repo_root=repo, bin_dir=home / ".local" / "bin")
    manager = FakeManager(repo)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(empty_bin))
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "run was not found in PATH" in result.output
    assert 'export PATH="$HOME/.local/bin:$PATH"' in result.output
    assert "make install-global" in result.output


def test_doctor_reports_ok_when_path_contains_april_wrapper(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    local_bin = home / ".local" / "bin"
    install_wrappers(repo_root=repo, bin_dir=local_bin)
    manager = FakeManager(repo)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(local_bin))
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "doctor"])
    assert result.exit_code == 0
    assert "OK: run resolves to an APRIL wrapper visible in PATH" in result.output


def test_doctor_reports_non_april_run_command(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    local_bin = home / ".local" / "bin"
    home.mkdir()
    repo.mkdir()
    local_bin.mkdir(parents=True)
    other_run = local_bin / "run"
    other_run.write_text("#!/usr/bin/env bash\necho other\n", encoding="utf-8")
    other_run.chmod(0o755)
    manager = FakeManager(repo)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(local_bin))
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "run resolves to a non-APRIL command" in result.output
    assert "make install-global-force" in result.output
