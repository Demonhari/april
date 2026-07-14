from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from apps.cli.main import _handle_repl_command, app


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
        if path == "/sessions":
            return {"sessions": [{"id": "session-1"}]}
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
        if path == "/playbooks":
            return {"playbooks": []}
        if path == "/evolution/versions":
            return {"versions": []}
        if path == "/evolution/report/latest":
            return {"report": None}
        if path == "/evolution/status":
            return {"status": {"enabled": False, "kill_switch_active": False}}
        if path == "/evolution/history":
            return {"runs": []}
        if path == "/evolution/diff":
            return {"agent": params["agent"], "diff": "+new line", "from_version": 1}
        if path == "/evolution/overlays/pending":
            return {"pending": []}
        if path == "/evolution/evals/pending":
            return {"pending": []}
        if path.startswith("/evolution/evals/pending/"):
            return {"case": {"case_id": path.rsplit("/", 1)[-1]}}
        if path == "/pool/agents":
            return {"agents": []}
        if path == "/memory/inspect":
            return {"memories": [], "state": params.get("state")}
        if path == "/voice/doctor":
            return {"status": "disabled"}
        raise AssertionError(path)

    async def post(
        self, path: str, payload: dict[str, Any], *, auth: bool = True
    ) -> dict[str, Any]:
        self.calls.append(("POST", path, payload))
        if path == "/chat":
            return {"result": {"final_message": "answer", "pending_approval": None}}
        if path == "/sessions":
            return {
                "session_id": "session-1",
                "conversation_id": "conversation-1",
                "joined_existing": False,
            }
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
        if path == "/playbooks/adopt":
            return {"adopted": True, "id": payload["id"]}
        if path.endswith("/run") and path.startswith("/playbooks/"):
            return {"run": {"status": "completed"}}
        if path == "/evolution/rollback":
            return {"rollback": {"status": "applied", **payload}}
        if path == "/evolution/off":
            return {"kill_switch_active": True}
        if path == "/evolution/on":
            return {"kill_switch_active": False}
        if path == "/evolution/dataset/export":
            return {
                "export": {
                    "path": "dataset.jsonl",
                    "chat_pairs": 0,
                    "preference_pairs": 0,
                }
            }
        if path == "/evolution/overlays/approve":
            return {"approval": {"status": "applied", **payload}}
        if path == "/evolution/evals/promote":
            return {"promoted": {"status": "reviewed", **payload}}
        if path == "/evolution/evals/reject":
            return {"rejected": {"status": "rejected", **payload}}
        if path.startswith("/sessions/") and path.endswith("/close"):
            return {"closed": True}
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
        ["agent", "pool"],
        ["projects"],
        ["project", "add", "/tmp/project"],
        ["project", "index", "project-1"],
        ["memory", "list"],
        ["memory", "search", "query"],
        ["memory", "delete", "memory-1"],
        ["memory", "export"],
        ["memory", "inspect", "--state", "superseded"],
        ["conversation", "delete", "conversation-1"],
        ["reminder", "list"],
        ["reminder", "create", "stand up", "--due-at", "2026-06-21T09:00:00Z"],
        ["reminder", "delete", "reminder-1"],
        ["task", "list"],
        ["playbook", "list"],
        ["playbook", "run", "sample"],
        ["evolve", "versions"],
        ["evolve", "rollback", "general_agent", "1"],
        ["evolve", "report"],
        ["evolve", "status"],
        ["evolve", "history"],
        ["evolve", "diff", "general_agent"],
        ["evolve", "off"],
        ["evolve", "on"],
        ["evolve", "pending"],
        ["evolve", "approve", "coding_agent", "a" * 64],
        ["evolve", "dataset", "export", "--name", "nightly"],
        ["evolve", "evals", "pending"],
        ["evolve", "evals", "show", "abc123"],
        ["evolve", "evals", "promote", "abc123", "--expected", "answer with timezone"],
        ["evolve", "evals", "reject", "abc123", "--reason", "not reproducible"],
        ["voice", "health"],
        ["voice", "doctor"],
    ]
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
    assert (
        "POST",
        "/chat",
        {
            "message": "hello",
            "project_id": None,
            "repo_path": None,
            "conversation_id": None,
            "mode": "standard",
        },
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
    assert ("GET", "/playbooks", None) in fake.calls
    assert (
        "POST",
        "/playbooks/sample/run",
        {"project_id": None, "conversation_id": None},
    ) in fake.calls
    assert ("GET", "/evolution/versions", None) in fake.calls
    assert (
        "POST",
        "/evolution/rollback",
        {"agent": "general_agent", "version": 1},
    ) in fake.calls
    assert ("GET", "/evolution/status", None) in fake.calls
    assert ("GET", "/evolution/history", {"limit": 20}) in fake.calls
    assert ("GET", "/evolution/diff", {"agent": "general_agent"}) in fake.calls
    assert ("POST", "/evolution/off", {}) in fake.calls
    assert ("POST", "/evolution/on", {}) in fake.calls
    assert ("GET", "/evolution/overlays/pending", None) in fake.calls
    assert (
        "POST",
        "/evolution/overlays/approve",
        {"agent": "coding_agent", "content_hash": "a" * 64},
    ) in fake.calls
    assert ("POST", "/evolution/dataset/export", {"name": "nightly"}) in fake.calls
    assert ("GET", "/evolution/evals/pending", None) in fake.calls
    assert ("GET", "/evolution/evals/pending/abc123", None) in fake.calls
    assert (
        "POST",
        "/evolution/evals/promote",
        {"case_id": "abc123", "expected_behavior": "answer with timezone"},
    ) in fake.calls
    assert (
        "POST",
        "/evolution/evals/reject",
        {"case_id": "abc123", "reason": "not reproducible"},
    ) in fake.calls


def test_playbook_adopt_reads_local_definition(monkeypatch, tmp_path) -> None:
    fake = FakeApiClient()
    monkeypatch.setattr("apps.cli.main.client", lambda: fake)
    path = tmp_path / "playbook.yaml"
    path.write_text(
        """
id: local-playbook
name: Local playbook
agent_id: general_agent
trigger_examples:
  - local
steps:
  - tool: create_reminder
    args:
      content: stand up
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["playbook", "adopt", str(path)])
    assert result.exit_code == 0, result.output
    assert fake.calls[-1][0:2] == ("POST", "/playbooks/adopt")
    assert fake.calls[-1][2]["id"] == "local-playbook"


def test_repl_slash_commands_delegate_to_existing_api(monkeypatch) -> None:
    fake = FakeApiClient()
    monkeypatch.setattr("apps.cli.main.client", lambda: fake)

    assert _handle_repl_command("/status", "conversation-1") is True
    assert ("GET", "/health", None) in fake.calls
    assert ("GET", "/sessions", None) in fake.calls
    assert ("GET", "/approvals", None) in fake.calls

    assert _handle_repl_command("/approve approval-1", "conversation-1") is True
    assert ("POST", "/tools/approve", {"approval_id": "approval-1"}) in fake.calls

    assert _handle_repl_command("/deny approval-1", "conversation-1") is True
    assert ("POST", "/tools/deny", {"approval_id": "approval-1"}) in fake.calls

    assert _handle_repl_command("/deep compare options", "conversation-1") is True
    assert (
        "POST",
        "/chat",
        {
            "message": "compare options",
            "conversation_id": "conversation-1",
            "mode": "deep",
        },
    ) in fake.calls

    assert _handle_repl_command("/council compare options", "conversation-1") is True
    assert (
        "POST",
        "/chat",
        {
            "message": "compare options",
            "conversation_id": "conversation-1",
            "mode": "council",
        },
    ) in fake.calls


def test_cli_announces_slow_modes_before_waiting(monkeypatch) -> None:
    fake = FakeApiClient()
    monkeypatch.setattr("apps.cli.main.client", lambda: fake)
    runner = CliRunner()

    deep = runner.invoke(app, ["ask", "compare options", "--mode", "deep"])
    assert deep.exit_code == 0, deep.output
    assert "Deep mode" in deep.output
    # Honest phrasing: no exact timing claim.
    assert "more carefully" in deep.output
    assert "answer" in deep.output

    council = runner.invoke(app, ["ask", "compare options", "--mode", "council"])
    assert council.exit_code == 0, council.output
    assert "Council mode" in council.output

    # Standard mode stays quiet — existing one-shot output is unchanged.
    standard = runner.invoke(app, ["ask", "hello"])
    assert standard.exit_code == 0, standard.output
    assert "Deep mode" not in standard.output
    assert "Council mode" not in standard.output


def test_voice_listen_attaches_to_resident_sentinel(monkeypatch) -> None:
    called: dict[str, object] = {}

    class Attachment:
        def __init__(self) -> None:
            self.status = {"ok": True, "state": "listening"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            called["closed"] = True

    def fake_attach(settings: object, *, session_hint: str):
        called["settings"] = settings
        called["session_hint"] = session_hint
        return Attachment()

    fake = FakeApiClient()
    monkeypatch.setattr("apps.cli.main.client", lambda: fake)
    monkeypatch.setattr("apps.cli.main._maybe_autostart_daemon", lambda: None)
    monkeypatch.setattr("services.wake.control.attach_resident_sentinel", fake_attach)
    monkeypatch.setattr("builtins.input", lambda: "")
    runner = CliRunner()
    result = runner.invoke(app, ["voice", "listen"])
    assert result.exit_code == 0, result.output
    assert called["session_hint"] == "session-1"
    assert called["closed"] is True


def test_top_level_listen_flag_uses_terminal_session_handoff(monkeypatch) -> None:
    fake = FakeApiClient()
    called: dict[str, object] = {}

    class Attachment:
        def __init__(self) -> None:
            self.status = {"ok": True, "state": "listening"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_attach(settings: object, *, session_hint: str):
        called["settings"] = settings
        called["session_hint"] = session_hint
        return Attachment()

    monkeypatch.setattr("apps.cli.main.client", lambda: fake)
    monkeypatch.setattr("apps.cli.main._maybe_autostart_daemon", lambda: None)
    monkeypatch.setattr("services.wake.control.attach_resident_sentinel", fake_attach)
    monkeypatch.setattr("builtins.input", lambda: "")
    result = CliRunner().invoke(app, ["--listen"])
    assert result.exit_code == 0, result.output
    assert "settings" in called
    assert called["session_hint"] == "session-1"
    assert ("POST", "/sessions", {"source": "terminal"}) in fake.calls
    assert ("POST", "/sessions/session-1/close", {}) in fake.calls


def test_speaker_gate_supports_off_and_soft() -> None:
    import pytest

    from april_common.settings import WakeSettings

    assert WakeSettings().speaker_gate == "off"
    # YAML 1.1 parses an unquoted `off` as False; it must still mean "off".
    assert WakeSettings(speaker_gate=False).speaker_gate == "off"
    assert WakeSettings(speaker_gate="soft").speaker_gate == "soft"
    with pytest.raises(ValueError, match="speaker_gate must be off or soft"):
        WakeSettings(speaker_gate="hard")


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
