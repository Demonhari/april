from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx

from apps.runner.verify import LauncherVerifier, VerifyCheck


def verifier_with_ports(monkeypatch) -> LauncherVerifier:
    ports = iter([18001, 18002])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    return LauncherVerifier(home=Path.cwd())


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
            if json["message"].startswith("Start a separate"):
                return FakeResponse({"result": {"status": "ok", "conversation_id": "conv-2"}})
            return FakeResponse({"result": {"status": "ok", "conversation_id": "conv-1"}})
        if path == "/tools/request":
            if "escape.patch" in str(json["args"]["patch_path"]):
                return FakeResponse({"error": {"message": "denied"}}, status_code=403)
            return FakeResponse({"approval": {"approval_id": "approval-1"}})
        if path == "/tools/approve":
            if json["approval_id"] == "approval-1":
                (self.verifier.project / "app.py").write_text("value = 'new'\n", encoding="utf-8")
                return FakeResponse({"status": "executed"})
            return FakeResponse({"error": {"message": "denied"}}, status_code=403)
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
    monkeypatch.setattr("apps.runner.verify.httpx.Client", lambda **kwargs: FakeClient(verifier))
    monkeypatch.setattr("apps.runner.verify.httpx.stream", lambda *args, **kwargs: FakeStream())

    assert verifier._check_models() == "1 models"
    assert verifier._register_project() == "project-1"
    assert verifier._multi_turn("project-1") == "conv-1"
    assert verifier._isolated_conversation("project-1", "conv-1") == "conv-2"
    assert verifier._repo_analysis("project-1") == "ok"
    approval_id = verifier._patch_approval("project-1")
    assert approval_id == "approval-1"
    assert verifier._approve(approval_id) == "applied"
    assert verifier._approval_replay_rejected("other") == "403"
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
    monkeypatch.setattr(verifier, "_patch_approval", lambda project_id: "approval")
    monkeypatch.setattr(verifier, "_approve", lambda approval_id: "applied")
    monkeypatch.setattr(verifier, "_approval_replay_rejected", lambda approval_id: "403")
    monkeypatch.setattr(verifier, "_tampered_artifact_rejected", lambda project_id: "failed")
    monkeypatch.setattr(verifier, "_path_escape_rejected", lambda project_id: "403")
    monkeypatch.setattr(verifier, "_repo_override_rejected", lambda: "403")
    monkeypatch.setattr(verifier, "_run_command_cwd_forced", lambda project_id: "forced")
    monkeypatch.setattr(verifier, "_runtime_streaming", lambda: "streamed")
    monkeypatch.setattr(verifier, "_audit_records", lambda: "audit")
    monkeypatch.setattr(verifier, "_tool_call_records", lambda: "1")

    checks = verifier.run()
    assert checks
    assert all(isinstance(check, VerifyCheck) for check in checks)
    assert all(check.ok for check in checks)
