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
        raise AssertionError(path)

    async def post(
        self, path: str, payload: dict[str, Any], *, auth: bool = True
    ) -> dict[str, Any]:
        self.calls.append(("POST", path, payload))
        if path == "/chat":
            return {"result": {"final_message": "answer", "pending_approval": None}}
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
        ["projects"],
        ["project", "add", "/tmp/project"],
        ["project", "index", "project-1"],
        ["memory", "list"],
        ["memory", "search", "query"],
        ["memory", "delete", "memory-1"],
        ["memory", "export"],
        ["conversation", "delete", "conversation-1"],
    ]
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
    assert (
        "POST",
        "/chat",
        {"message": "hello", "project_id": None, "repo_path": None, "conversation_id": None},
    ) in fake.calls
    assert ("DELETE", "/conversations/conversation-1", None) in fake.calls
