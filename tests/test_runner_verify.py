from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from apps.runner.evals import BrainEvalCase
from apps.runner.verify import (
    AllConfiguredModelsVerifier,
    LauncherVerifier,
    ModelBenchmark,
    RealModelVerifier,
    RealWorkflowVerifier,
    TargetMacValidator,
    VerifyCheck,
    WorkflowVerifier,
    brain_decision_after_marker,
    build_workflow_report,
    latest_brain_decision_marker,
    run_model_benchmark,
    run_real_model_verification,
    run_target_mac_validation,
    run_workflow_verification,
    write_workflow_report,
)


def verifier_with_ports(monkeypatch) -> LauncherVerifier:
    ports = iter([18001, 18002])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    return LauncherVerifier(home=Path.cwd())


def test_all_configured_specialist_smoke_schema_validators() -> None:
    verifier = object.__new__(AllConfiguredModelsVerifier)
    coding_prompt, coding_kind, coding_validator = verifier._smoke_spec("coding")
    _system_prompt, system_kind, system_validator = verifier._smoke_spec("system_action")
    reading_prompt, reading_kind, reading_validator = verifier._smoke_spec("reading")

    assert "JSON" in coding_prompt
    assert coding_kind == "coding_plan"
    assert coding_validator is not None
    assert coding_validator('{"plan":["edit","test"]}') is True
    assert coding_validator('```json\n{"plan":["edit","test"]}\n```') is True
    assert coding_validator('<think>short</think>\n{"plan":["edit","test"]}') is True
    assert coding_validator('{"plan":"edit"}') is False
    assert system_kind == "system_decision"
    assert system_validator is not None
    assert system_validator('{"execute":false,"permission_level":0}') is True
    assert system_validator('Result: {"execute":false,"permission_level":0}') is True
    assert system_validator('{"execute":true,"permission_level":5}') is False
    assert reading_kind == "reading_summary"
    assert reading_validator is None
    assert "summarize" in reading_prompt.lower()


def test_all_configured_prepare_resolves_relative_model_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    verify_home = tmp_path / "verify_home"
    (home / "configs").mkdir(parents=True)
    (home / "configs" / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "models": {
                    "brain": {
                        "id": "april-brain",
                        "name": "brain",
                        "path": "models/brain.gguf",
                        "backend": "llama_cpp",
                        "role": "brain",
                        "threads": 1,
                        "context_size": 1024,
                        "temperature": 0.0,
                        "max_output_tokens": 16,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    verifier = object.__new__(AllConfiguredModelsVerifier)
    verifier.repo_home = home
    verifier.verify_home = verify_home

    verifier._prepare()

    copied = yaml.safe_load((verify_home / "configs" / "models.yaml").read_text())
    assert copied["models"]["brain"]["path"] == str(home / "models" / "brain.gguf")


class FakeResponse:
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)  # type: ignore[arg-type]


class FakeClient:
    def __init__(self, verifier: LauncherVerifier) -> None:
        self.verifier = verifier

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, path: str) -> FakeResponse:
        if path == "/runtime/models":
            return FakeResponse({"models": [{"id": "april-brain"}]})
        raise AssertionError(path)

    def post(self, path: str, json: dict[str, Any]) -> FakeResponse:
        if path == "/projects":
            return FakeResponse({"id": "project-1"})
        if path == "/chat":
            if json["message"] == "Apply the fix.":
                count = getattr(self.verifier, "_fake_approval_count", 0) + 1
                self.verifier._fake_approval_count = count
                return FakeResponse(
                    {
                        "result": {
                            "status": "pending_approval",
                            "conversation_id": "conv-3",
                            "pending_approval": {
                                "approval_id": f"approval-{count}",
                                "metadata": {"agent_run_id": "run-1"},
                            },
                        }
                    }
                )
            if json["message"].startswith("Start a separate"):
                return FakeResponse({"result": {"status": "ok", "conversation_id": "conv-2"}})
            return FakeResponse({"result": {"status": "ok", "conversation_id": "conv-1"}})
        if path == "/agents/run":
            return FakeResponse({"result": {"status": "ok", "conversation_id": "conv-agent"}})
        if path == "/tools/request":
            if "escape.patch" in str(json["args"]["patch_path"]):
                return FakeResponse({"error": {"message": "denied"}}, status_code=403)
            return FakeResponse({"approval": {"approval_id": "approval-1"}})
        if path == "/tools/approve":
            approved = getattr(self.verifier, "_fake_approved_ids", set())
            if json["approval_id"] in approved:
                return FakeResponse({"error": {"message": "denied"}}, status_code=403)
            if json["approval_id"] == "approval-3":
                (self.verifier.project / "README.md").write_text(
                    "# verify\nanimation bug\nfixed animation\n",
                    encoding="utf-8",
                )
                approved.add(json["approval_id"])
                self.verifier._fake_approved_ids = approved
                return FakeResponse({"status": "resumed", "result": {"status": "ok"}})
            return FakeResponse({"error": {"message": "denied"}}, status_code=403)
        if path == "/tools/deny":
            return FakeResponse({"status": "denied"})
        raise AssertionError(path)


class FakeStream:
    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return ["event: token", "data: {}", "event: usage", "data: {}", "event: done"]


def test_verifier_helpers_use_temp_environment(tmp_path: Path, monkeypatch) -> None:
    verifier = verifier_with_ports(monkeypatch)
    try:
        assert verifier.runtime_url.startswith("http://127.0.0.1:")
        assert verifier.api_url.startswith("http://127.0.0.1:")
        verifier._prepare()
        env = verifier._env()
        assert env["APRIL_HOME"] == str(verifier.verify_home)
        assert env["APRIL_RUNTIME_BACKEND"] == "fake"
        assert str(verifier.project) in env["APRIL_ALLOWED_FILESYSTEM_ROOTS"]
        assert str(verifier.second_project) in env["APRIL_ALLOWED_FILESYSTEM_ROOTS"]
    finally:
        verifier._stop()


def test_verifier_http_steps_are_deterministic(tmp_path: Path, monkeypatch) -> None:
    verifier = verifier_with_ports(monkeypatch)
    verifier.verify_home.mkdir(parents=True, exist_ok=True)
    (verifier.verify_home / "data" / "patches").mkdir(parents=True, exist_ok=True)
    verifier.project.mkdir(parents=True, exist_ok=True)
    (verifier.project / "app.py").write_text("value = 'old'\n", encoding="utf-8")
    (verifier.project / "README.md").write_text("# verify\nanimation bug\n", encoding="utf-8")
    monkeypatch.setattr("apps.runner.verify.httpx.Client", lambda **kwargs: FakeClient(verifier))
    monkeypatch.setattr("apps.runner.verify.httpx.stream", lambda *args, **kwargs: FakeStream())

    assert verifier._check_models() == "1 models"
    assert verifier._register_project() == "project-1"
    assert verifier._multi_turn("project-1") == "conv-1"
    assert verifier._isolated_conversation("project-1", "conv-1") == "conv-2"
    assert verifier._repo_analysis("project-1") == "ok"
    assert verifier._direct_agent_run("project-1") == "ok"
    denial_approval_id = verifier._patch_approval("project-1")
    assert denial_approval_id == "approval-1"
    assert verifier._deny_approval(denial_approval_id) == "denied"
    expired_approval_id = verifier._patch_approval("project-1")
    assert expired_approval_id == "approval-2"
    assert verifier._expired_approval_rejected(expired_approval_id) == "403 expired"
    approval_id = verifier._patch_approval("project-1")
    assert approval_id == "approval-3"
    assert verifier._approve(approval_id) == "applied and resumed"
    assert verifier._approval_replay_rejected(approval_id) == "403"
    assert verifier._path_escape_rejected("project-1") == "403"
    assert "token events" in verifier._runtime_streaming()
    audit = verifier.temp / "logs" / "audit.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("approval_consumed\napproved_tool_executed\n", encoding="utf-8")
    assert verifier._audit_records() == "ok"


def test_verifier_run_records_failures_and_stops(tmp_path: Path, monkeypatch) -> None:
    verifier = verifier_with_ports(monkeypatch)
    monkeypatch.setattr(verifier, "_prepare", lambda: None)
    monkeypatch.setattr(verifier, "_env", lambda: {})
    monkeypatch.setattr(verifier, "_start", lambda *args: subprocess.Popen(["true"]))
    monkeypatch.setattr(verifier, "_wait_json", lambda url: {"status": "ok"})
    monkeypatch.setattr(verifier, "_check_models", lambda: "models")
    monkeypatch.setattr(verifier, "_register_project", lambda: "project")
    monkeypatch.setattr(verifier, "_multi_turn", lambda project_id: "conversation")
    monkeypatch.setattr(verifier, "_isolated_conversation", lambda project_id, conv: "other")
    monkeypatch.setattr(verifier, "_conversation_switch_rejected", lambda conv: "403")
    monkeypatch.setattr(verifier, "_repo_analysis", lambda project_id: "ok")
    monkeypatch.setattr(verifier, "_direct_agent_run", lambda project_id: "ok")
    monkeypatch.setattr(verifier, "_patch_approval", lambda project_id: "approval")
    monkeypatch.setattr(verifier, "_deny_approval", lambda approval_id: "denied")
    monkeypatch.setattr(verifier, "_expired_approval_rejected", lambda approval_id: "403 expired")
    monkeypatch.setattr(verifier, "_approve", lambda approval_id: "applied")
    monkeypatch.setattr(verifier, "_approval_replay_rejected", lambda approval_id: "403")
    monkeypatch.setattr(verifier, "_tampered_artifact_rejected", lambda project_id: "failed")
    monkeypatch.setattr(verifier, "_path_escape_rejected", lambda project_id: "403")
    monkeypatch.setattr(verifier, "_repo_override_rejected", lambda: "403")
    monkeypatch.setattr(verifier, "_run_command_cwd_forced", lambda project_id: "forced")
    monkeypatch.setattr(verifier, "_runtime_streaming", lambda: "streamed")
    monkeypatch.setattr(verifier, "_audit_records", lambda: "audit")
    monkeypatch.setattr(verifier, "_tool_call_records", lambda: "1")
    monkeypatch.setattr(verifier, "_agent_run_records", lambda: "runs")

    checks = verifier.run()
    assert checks
    assert all(isinstance(check, VerifyCheck) for check in checks)
    assert all(check.ok for check in checks)


def test_real_model_verifier_runs_load_chat_stream_unload_and_stop(
    tmp_path: Path, monkeypatch
) -> None:
    ports = iter([19001, 19002])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    verifier = RealModelVerifier(home=Path.cwd(), model_path=tmp_path / "model.gguf")
    calls: list[str] = []
    monkeypatch.setattr(verifier, "_prepare", lambda: calls.append("prepare"))
    monkeypatch.setattr(verifier, "_env", lambda: {})
    monkeypatch.setattr(verifier, "_start", lambda *args: subprocess.Popen(["true"]))
    monkeypatch.setattr(verifier, "_wait_json", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(verifier, "_load_model", lambda: calls.append("load") or "loaded")
    monkeypatch.setattr(verifier, "_chat", lambda: calls.append("chat") or "ok")
    monkeypatch.setattr(verifier, "_stream", lambda: calls.append("stream") or "tokens")
    monkeypatch.setattr(verifier, "_unload_model", lambda: calls.append("unload") or "unloaded")
    monkeypatch.setattr(
        verifier, "_confirm_unloaded", lambda: calls.append("confirm") or "unloaded"
    )

    checks = verifier.run()
    assert calls == ["prepare", "load", "chat", "stream", "unload", "confirm"]
    assert all(check.ok for check in checks)
    assert checks[-1].name == "services stopped"


def test_real_model_verifier_prepare_and_env(tmp_path: Path, monkeypatch) -> None:
    ports = iter([19101, 19102])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    verifier = RealModelVerifier(home=Path.cwd(), model_path=gguf)
    try:
        verifier._prepare()
        models_text = (verifier.verify_home / "configs" / "models.yaml").read_text(encoding="utf-8")
        assert "april-brain" in models_text
        assert "april-coding" in models_text
        assert str(gguf) in models_text
        env = verifier._env()
        assert env["APRIL_RUNTIME_BACKEND"] == "llama_cpp"
        assert env["APRIL_RUNTIME_PRELOAD_KEEP_LOADED"] == "false"
        assert env["APRIL_RUNTIME_TOKEN"] == verifier.runtime_token
        assert env["APRIL_RUNTIME_URL"] == verifier.runtime_url
    finally:
        shutil.rmtree(verifier.temp, ignore_errors=True)


def test_run_real_model_verification_uses_real_verifier(tmp_path: Path, monkeypatch) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: True)

    class FakeRealModelVerifier:
        def __init__(
            self,
            *,
            home: Path,
            model_path: Path,
            max_output_tokens: int = 32,
            timeout: float = 180.0,
        ) -> None:
            self.home = home
            self.model_path = model_path

        def run(self) -> list[VerifyCheck]:
            return [VerifyCheck(name=str(self.model_path), ok=True, detail=str(self.home))]

    monkeypatch.setattr("apps.runner.verify.RealModelVerifier", FakeRealModelVerifier)
    checks = run_real_model_verification(Path.cwd(), gguf)
    assert checks == [VerifyCheck(name=str(gguf), ok=True, detail=str(Path.cwd()))]


def test_run_real_model_verification_reports_runtime_extra_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: False)
    checks = run_real_model_verification(Path.cwd(), gguf)
    assert checks[0].ok is False
    assert checks[0].detail == "pip install -e '.[runtime]'"


def test_run_model_benchmark_reports_runtime_extra_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: False)
    results = run_model_benchmark(
        Path.cwd(),
        gguf,
        prompt="hello",
        runs=1,
        max_output_tokens=4,
        keep_loaded=False,
    )
    assert results[0].ok is False
    assert "pip install -e" in results[0].detail


def test_target_mac_validation_skips_optional_model_and_voice(monkeypatch) -> None:
    monkeypatch.setattr("apps.runner.verify.platform.system", lambda: "Darwin")
    monkeypatch.setattr("apps.runner.verify.platform.machine", lambda: "arm64")
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: False)
    monkeypatch.setattr(
        "apps.runner.verify.query_audio_devices",
        lambda: {
            "sounddevice_installed": False,
            "input_devices": [],
            "output_devices": [],
            "error": "missing",
        },
    )
    checks = run_target_mac_validation(Path.cwd())
    assert all(check.ok for check in checks)
    assert any(
        check.name == "llama-cpp-python import" and check.status == "skip" for check in checks
    )
    assert any(check.name == "configured GGUF existence and readability" for check in checks)
    assert any(check.status == "manual" for check in checks)


def test_target_mac_validation_can_require_real_model(monkeypatch) -> None:
    monkeypatch.setattr("apps.runner.verify.platform.system", lambda: "Darwin")
    monkeypatch.setattr("apps.runner.verify.platform.machine", lambda: "arm64")
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: False)
    checks = run_target_mac_validation(Path.cwd(), require_real_model=True)
    assert any(check.name == "llama-cpp-python import" and check.ok is False for check in checks)
    assert not all(check.ok for check in checks)


def test_target_mac_validation_runs_real_model_checks_when_ready(
    tmp_path: Path, monkeypatch
) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    fake_llama = types.SimpleNamespace(
        __version__="test",
        llama_print_system_info=lambda: b"metal=1",
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)
    monkeypatch.setattr("apps.runner.verify.platform.system", lambda: "Darwin")
    monkeypatch.setattr("apps.runner.verify.platform.machine", lambda: "arm64")
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: True)
    monkeypatch.setattr(
        "apps.runner.verify.query_audio_devices",
        lambda: {
            "sounddevice_installed": True,
            "input_devices": [{"index": 0, "name": "Mic"}],
            "output_devices": [{"index": 1, "name": "Speaker"}],
        },
    )
    monkeypatch.setattr(
        "apps.runner.verify.voice_doctor",
        lambda settings: {
            "components": [
                {"name": "whisper binary", "status": "ok", "message": "whisper"},
                {"name": "whisper model", "status": "ok", "message": "model"},
                {"name": "piper binary", "status": "ok", "message": "piper"},
                {"name": "piper model", "status": "ok", "message": "voice"},
                {"name": "wake-word model", "status": "ok", "message": "wake"},
            ]
        },
    )

    class _StubRealVerifier:
        # TargetMacValidator now instantiates RealModelVerifier directly so it can
        # read structured metrics for the report; the stub provides both.
        def __init__(self, **kwargs: object) -> None:
            self.load_time_seconds = 1.0
            self.first_token_latency_seconds = 0.2
            self.generation_time_seconds = 1.0
            self.output_tokens = 8
            self.tokens_per_second = 8.0
            self.runtime_rss_bytes = 1000

        def run(self) -> list[VerifyCheck]:
            return [VerifyCheck(name="real model load", ok=True, detail="loaded")]

    monkeypatch.setattr("apps.runner.verify.RealModelVerifier", _StubRealVerifier)
    monkeypatch.setattr(
        "apps.runner.verify.run_workflow_verification",
        lambda home, **kwargs: [
            VerifyCheck(name="real workflow specialist-agent request", ok=True, detail="ok")
        ],
    )
    validator = TargetMacValidator(
        home=Path.cwd(),
        model_path=gguf,
        require_real_model=False,
        max_output_tokens=8,
        timeout=5.0,
    )
    checks = validator.run()
    names = {check.name for check in checks}
    assert "real model load" in names
    assert "real workflow specialist-agent request" in names
    assert any(check.name == "backend acceleration/build information" for check in checks)
    assert all(check.ok for check in checks)
    # The report captures the real-model metrics from the verifier instance.
    report = validator.build_report()
    assert report.real_model.attempted is True
    assert report.real_model.tokens_per_second == 8.0
    assert report.real_model.path_basename == "model.gguf"


def test_model_benchmark_run_and_single_success(tmp_path: Path, monkeypatch) -> None:
    ports = iter([19301, 19302])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    benchmark = ModelBenchmark(
        home=Path.cwd(),
        model_path=tmp_path / "model.gguf",
        prompt="hello",
        runs=2,
        max_output_tokens=4,
        keep_loaded=False,
    )
    monkeypatch.setattr(benchmark, "_prepare", lambda: None)
    monkeypatch.setattr(benchmark, "_env", lambda: {})
    monkeypatch.setattr(benchmark, "_start", lambda *args: subprocess.Popen(["true"]))
    monkeypatch.setattr(benchmark, "_wait_json", lambda *args, **kwargs: {"status": "ok"})
    calls: list[int] = []
    monkeypatch.setattr(
        benchmark,
        "_run_one",
        lambda index: (
            calls.append(index)
            or __import__("apps.runner.verify", fromlist=["BenchmarkResult"]).BenchmarkResult(
                run_index=index,
                ok=True,
            )
        ),
    )
    results = benchmark.run()
    assert [result.run_index for result in results] == [1, 2]
    assert calls == [1, 2]


def test_model_benchmark_run_one_unloads(tmp_path: Path, monkeypatch) -> None:
    ports = iter([19401, 19402])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    benchmark = ModelBenchmark(
        home=Path.cwd(),
        model_path=tmp_path / "model.gguf",
        prompt="hello",
        runs=1,
        max_output_tokens=4,
        keep_loaded=False,
    )
    calls: list[str] = []
    benchmark.load_time_seconds = 1.0
    benchmark.generation_time_seconds = 2.0
    benchmark.first_token_latency_seconds = 0.5
    benchmark.output_tokens = 6
    benchmark.tokens_per_second = 3.0
    monkeypatch.setattr(benchmark, "_load_model", lambda: calls.append("load") or "loaded")
    monkeypatch.setattr(benchmark, "_benchmark_stream", lambda: calls.append("stream"))
    monkeypatch.setattr(benchmark, "_unload_model", lambda: calls.append("unload") or "unloaded")
    result = benchmark._run_one(1)
    assert result.ok is True
    assert result.unload_success is True
    assert calls == ["load", "stream", "unload"]


def test_workflow_verifier_run_uses_release_checklist(monkeypatch) -> None:
    ports = iter([19601, 19602])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    workflow = WorkflowVerifier(home=Path.cwd())
    calls: list[str] = []
    monkeypatch.setattr(workflow, "_prepare", lambda: calls.append("prepare"))
    monkeypatch.setattr(workflow, "_env", lambda: {})
    monkeypatch.setattr(workflow, "_start", lambda *args: subprocess.Popen(["true"]))
    monkeypatch.setattr(workflow, "_wait_json", lambda url: {"status": "ok"})
    monkeypatch.setattr(workflow, "_model_load_unload", lambda: "loaded -> unloaded")
    monkeypatch.setattr(workflow, "_register_project", lambda: "project")
    monkeypatch.setattr(workflow, "_multi_turn", lambda project_id: "conversation")
    monkeypatch.setattr(workflow, "_task_listing", lambda: "1 tasks")
    monkeypatch.setattr(workflow, "_repo_analysis", lambda project_id: "ok")
    monkeypatch.setattr(workflow, "_patch_approval", lambda project_id: "approval")
    monkeypatch.setattr(workflow, "_deny_approval", lambda approval_id: "denied")
    monkeypatch.setattr(workflow, "_approve", lambda approval_id: "applied")
    monkeypatch.setattr(workflow, "_system_action_policy", lambda: "checked")
    monkeypatch.setattr(workflow, "_reminder_create_list", lambda: "1 reminders")
    monkeypatch.setattr(workflow, "_voice_health", lambda: "disabled")
    checks = workflow.run()
    assert calls == ["prepare"]
    assert all(check.ok for check in checks)
    assert any(check.name == "voice health" for check in checks)


def test_run_workflow_verification_passes_real_model_settings(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _StubRealWorkflowVerifier:
        def __init__(
            self,
            *,
            home: Path,
            model_path: Path,
            max_output_tokens: int,
            timeout: float,
        ) -> None:
            captured.update(
                {
                    "home": home,
                    "model_path": model_path,
                    "max_output_tokens": max_output_tokens,
                    "timeout": timeout,
                }
            )

        def run(self) -> list[VerifyCheck]:
            return [VerifyCheck(name="real workflow planning route", ok=True, detail="ok")]

    monkeypatch.setattr("apps.runner.verify.RealWorkflowVerifier", _StubRealWorkflowVerifier)
    checks = run_workflow_verification(
        tmp_path,
        real_model=True,
        model_path=tmp_path / "model.gguf",
        max_output_tokens=13,
        timeout=7.5,
    )
    assert checks[0].ok is True
    assert captured["max_output_tokens"] == 13
    assert captured["timeout"] == 7.5


def test_workflow_report_redacts_sensitive_details(tmp_path: Path) -> None:
    report = build_workflow_report(
        [
            VerifyCheck(
                name="real workflow planning route",
                ok=True,
                detail=(
                    "prompt private /Users/hari/secret/project token abc "
                    "transcript words decision_summary raw_tool_args"
                ),
            )
        ],
        real_model_requested=True,
        timeout_seconds=7.5,
        max_output_tokens=13,
    )
    out = tmp_path / "workflow.json"
    write_workflow_report(report, out)
    payload = out.read_text(encoding="utf-8")
    assert "timeout_seconds" in payload
    assert "max_output_tokens" in payload
    for secret in (
        "prompt private",
        "/Users/hari/secret/project",
        "token abc",
        "transcript words",
        "decision_summary raw_tool_args",
    ):
        assert secret not in payload


def test_fake_workflow_report_omits_real_verifier_settings(tmp_path: Path) -> None:
    report = build_workflow_report(
        [VerifyCheck(name="workflow voice health", ok=True, detail="disabled")],
        real_model_requested=False,
        timeout_seconds=7.5,
        max_output_tokens=13,
    )
    out = tmp_path / "workflow-fake.json"
    write_workflow_report(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["real_model_verified"] is False
    assert "timeout_seconds" not in payload
    assert "max_output_tokens" not in payload


def test_real_workflow_latest_routing_method(tmp_path: Path, monkeypatch) -> None:
    ports = iter([19501, 19502])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    verifier = RealWorkflowVerifier(home=Path.cwd(), model_path=tmp_path / "model.gguf")
    try:
        database = verifier.temp / "data" / "april.db"
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as conn:
            conn.execute(
                "CREATE TABLE conversation_events("
                "event_type TEXT, payload_json TEXT, created_at TEXT)"
            )
            conn.execute(
                "INSERT INTO conversation_events VALUES(?, ?, ?)",
                ("brain_decision", '{"routing_method":"model"}', "2026-01-01T00:00:00Z"),
            )
        assert verifier._latest_routing_method() == "model"
    finally:
        shutil.rmtree(verifier.temp, ignore_errors=True)


def test_real_workflow_run_uses_expanded_safe_checklist(tmp_path: Path, monkeypatch) -> None:
    ports = iter([19531, 19532])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    verifier = RealWorkflowVerifier(home=Path.cwd(), model_path=tmp_path / "model.gguf")
    monkeypatch.setattr(verifier, "_prepare", lambda: None)
    monkeypatch.setattr(verifier, "_env", lambda: {})
    monkeypatch.setattr(verifier, "_start", lambda *args: subprocess.Popen(["true"]))
    monkeypatch.setattr(verifier, "_wait_json", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(verifier, "_real_planning_route", lambda: "routing_method=model")
    monkeypatch.setattr(verifier, "_real_specialist_agent", lambda: "reading_agent ok")
    monkeypatch.setattr(verifier, "_register_temp_project", lambda: "project")
    monkeypatch.setattr(verifier, "_real_reminder_create_list", lambda: "1 reminders")
    monkeypatch.setattr(verifier, "_real_memory_write_search", lambda: "1 memory results")
    monkeypatch.setattr(verifier, "_real_document_index_search", lambda: "1 document chunks")
    monkeypatch.setattr(verifier, "_real_coding_repo_analysis", lambda project_id: "coding ok")
    monkeypatch.setattr(verifier, "_real_code_write_approval", lambda project_id: "approval")
    monkeypatch.setattr(verifier, "_real_approval_denial", lambda approval_id: "denied")
    monkeypatch.setattr(verifier, "_real_external_action_denial", lambda: "403")
    monkeypatch.setattr(verifier, "_real_voice_health", lambda: "disabled")
    checks = verifier.run()
    names = {check.name for check in checks}
    for name in (
        "runtime health",
        "core health",
        "real workflow planning route",
        "real workflow specialist-agent request",
        "workflow project registration",
        "workflow reminder create/list",
        "workflow memory write/search",
        "workflow document indexing/search",
        "workflow coding read-only repo analysis",
        "workflow code-write approval creation",
        "workflow approval denial",
        "workflow external action denial",
        "workflow voice health",
    ):
        assert name in names
    assert all(check.ok for check in checks)


def _create_decision_table(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE conversation_events("
            "id TEXT PRIMARY KEY, event_type TEXT, payload_json TEXT, created_at TEXT)"
        )


def test_brain_decision_after_marker_does_not_reuse_stale_decision(tmp_path: Path) -> None:
    database = tmp_path / "april.db"
    _create_decision_table(database)
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO conversation_events VALUES(?, ?, ?, ?)",
            ("old", "brain_decision", '{"routing_method":"model"}', "2026-01-01T00:00:00Z"),
        )
    marker = latest_brain_decision_marker(database)
    assert marker > 0
    assert brain_decision_after_marker(database, marker) == {}


def test_brain_decision_after_marker_uses_new_decision(tmp_path: Path) -> None:
    database = tmp_path / "april.db"
    _create_decision_table(database)
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO conversation_events VALUES(?, ?, ?, ?)",
            ("old", "brain_decision", '{"routing_method":"fallback"}', "2026-01-01T00:00:00Z"),
        )
    marker = latest_brain_decision_marker(database)
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO conversation_events VALUES(?, ?, ?, ?)",
            ("new", "brain_decision", '{"routing_method":"model"}', "2026-01-01T00:00:01Z"),
        )
    assert brain_decision_after_marker(database, marker)["routing_method"] == "model"


def test_real_workflow_planning_route_fails_when_no_new_decision(
    tmp_path: Path, monkeypatch
) -> None:
    ports = iter([19521, 19522])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    verifier = RealWorkflowVerifier(home=Path.cwd(), model_path=tmp_path / "model.gguf")

    class PlanningClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> PlanningClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, path: str, json: dict[str, Any]) -> FakeResponse:
            assert path == "/chat"
            return FakeResponse({"result": {"status": "ok"}})

    monkeypatch.setattr("apps.runner.verify.httpx.Client", PlanningClient)
    try:
        with pytest.raises(RuntimeError, match="no new brain_decision"):
            verifier._real_planning_route()
    finally:
        shutil.rmtree(verifier.temp, ignore_errors=True)


def test_all_configured_routing_report_counts_missing_decisions(
    tmp_path: Path, monkeypatch
) -> None:
    verifier = object.__new__(AllConfiguredModelsVerifier)
    verifier.repo_home = Path.cwd()
    verifier.temp = tmp_path
    verifier.api_port = 19999
    verifier.api_token = "token"
    verifier.timeout = 1.0
    cases = [
        BrainEvalCase(
            id="c1",
            message="hello",
            expected_intent="normal_conversation",
            expected_agent="general_agent",
        )
    ]
    monkeypatch.setattr("apps.runner.evals.load_brain_eval_cases", lambda home: cases)

    class RoutingClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> RoutingClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, path: str, json: dict[str, Any]) -> FakeResponse:
            assert path == "/chat"
            return FakeResponse({"result": {"status": "ok"}})

    monkeypatch.setattr("apps.runner.verify.httpx.Client", RoutingClient)
    report = verifier._routing_report()
    assert report.total == 1
    assert report.passed == 0


def test_real_workflow_specialist_agent_request(tmp_path: Path, monkeypatch) -> None:
    ports = iter([19511, 19512])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    verifier = RealWorkflowVerifier(home=Path.cwd(), model_path=tmp_path / "model.gguf")

    class SpecialistClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> SpecialistClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, path: str, json: dict[str, Any]) -> FakeResponse:
            assert path == "/agents/run"
            assert json["agent"] == "reading_agent"
            return FakeResponse({"result": {"status": "ok", "conversation_id": "conv"}})

    monkeypatch.setattr("apps.runner.verify.httpx.Client", SpecialistClient)
    try:
        assert verifier._real_specialist_agent() == "reading_agent ok"
    finally:
        shutil.rmtree(verifier.temp, ignore_errors=True)


def test_real_model_verifier_response_error_and_stopped(tmp_path: Path, monkeypatch) -> None:
    ports = iter([19201, 19202])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    verifier = RealModelVerifier(home=Path.cwd(), model_path=tmp_path / "model.gguf")
    try:
        json_error = httpx.Response(
            503,
            json={"error": {"message": "load failed", "details": {"cause": "missing llama"}}},
        )
        assert "missing llama" in verifier._response_error(json_error)
        text_error = httpx.Response(500, text="plain failure")
        assert verifier._response_error(text_error) == "plain failure"
        assert verifier._services_stopped() == "stopped"
    finally:
        shutil.rmtree(verifier.temp, ignore_errors=True)
