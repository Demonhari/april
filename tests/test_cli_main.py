from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from apps.cli.main import app


class FakeApiClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []

    async def get(
        self, path: str, *, params: dict[str, Any] | None = None, auth: bool = True
    ) -> dict[str, Any]:
        self.calls.append(("GET", path, params))
        if path == "/health":
            return {"status": "ok"}
        if path == "/runtime/models":
            return {"models": [{"id": "april-brain", "role": "brain", "state": "loaded"}]}
        if path == "/approvals":
            return {"approvals": []}
        if path == "/projects":
            return {"projects": []}
        if path == "/memory/export":
            return {"export": "{}"}
        if path == "/memory/search":
            return {"results": []}
        if path == "/reminders":
            return {"reminders": []}
        if path == "/tasks":
            return {"tasks": []}
        if path == "/voice/doctor":
            return {"status": "disabled"}
        raise AssertionError(path)

    async def post(
        self, path: str, payload: dict[str, Any], *, auth: bool = True
    ) -> dict[str, Any]:
        self.calls.append(("POST", path, payload))
        if path == "/chat":
            return {"result": {"final_message": "answer", "pending_approval": None}}
        if path == "/agents/run":
            return {"result": {"final_message": "agent answer", "pending_approval": None}}
        if path == "/tools/approve":
            return {"status": "executed"}
        if path == "/tools/deny":
            return {"status": "denied"}
        if path.startswith("/runtime/models/"):
            return {"status": "ok"}
        if path == "/projects":
            return {"id": "project-1"}
        if path.endswith("/index"):
            return {"result": {"ok": True}}
        if path == "/reminders":
            return {"reminder": {"id": "reminder-1", **payload}}
        raise AssertionError(path)

    async def delete(self, path: str) -> dict[str, Any]:
        self.calls.append(("DELETE", path, None))
        return {"deleted": True}


def test_cli_commands_delegate_to_api(monkeypatch) -> None:
    fake = FakeApiClient()
    monkeypatch.setattr("apps.cli.main.client", lambda: fake)
    runner = CliRunner()
    commands = [
        ["health"],
        ["ask", "hello"],
        ["models"],
        ["model", "load", "april-brain"],
        ["model", "unload", "april-brain"],
        ["approvals"],
        ["approve", "approval-1"],
        ["deny", "approval-1"],
        ["agent", "run", "coding_agent", "inspect"],
        ["projects"],
        ["project", "add", "/tmp/project"],
        ["project", "index", "project-1"],
        ["memory", "list"],
        ["memory", "search", "query"],
        ["memory", "delete", "memory-1"],
        ["memory", "export"],
        ["conversation", "delete", "conversation-1"],
        ["reminder", "list"],
        ["reminder", "create", "stand up", "--due-at", "2026-06-21T09:00:00Z"],
        ["reminder", "delete", "reminder-1"],
        ["task", "list"],
        ["voice", "health"],
        ["voice", "doctor"],
    ]
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
    assert (
        "POST",
        "/chat",
        {"message": "hello", "project_id": None, "repo_path": None, "conversation_id": None},
    ) in fake.calls
    assert (
        "POST",
        "/agents/run",
        {
            "agent": "coding_agent",
            "message": "inspect",
            "project_id": None,
            "repo_path": None,
            "conversation_id": None,
            "options": {"structured": True},
        },
    ) in fake.calls
    assert ("DELETE", "/conversations/conversation-1", None) in fake.calls
    assert (
        "POST",
        "/reminders",
        {"content": "stand up", "due_at": "2026-06-21T09:00:00Z"},
    ) in fake.calls
    assert ("DELETE", "/reminders/reminder-1", None) in fake.calls


def test_voice_ptt_modes_use_capture_strategy(monkeypatch) -> None:
    import services.voice.conversation_loop as conversation_loop

    constructed: dict[str, Any] = {}

    class StubLoop:
        def __init__(self, **kwargs: Any) -> None:
            constructed.clear()
            constructed.update(kwargs)

        async def run_once(self) -> str:
            return "spoken answer"

    monkeypatch.setattr(conversation_loop, "PushToTalkLoop", StubLoop)
    monkeypatch.setattr("apps.cli.main.client", lambda: object())
    runner = CliRunner()

    # Fixed-duration (--seconds) mode passes record_seconds and no capture strategy.
    fixed = runner.invoke(app, ["voice", "ptt", "--seconds", "2"])
    assert fixed.exit_code == 0, fixed.output
    assert "spoken answer" in fixed.output
    assert constructed["record_seconds"] == 2.0
    assert constructed.get("capture") is None

    # Interactive mode injects a stop-controlled capture strategy and a microphone.
    interactive = runner.invoke(app, ["voice", "ptt"])
    assert interactive.exit_code == 0, interactive.output
    assert constructed.get("capture") is not None
    assert constructed.get("microphone") is not None
