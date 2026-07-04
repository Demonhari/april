from __future__ import annotations

import json
from datetime import UTC, datetime

import anyio
import pytest
from fastapi.testclient import TestClient

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.api.server import create_app
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.scheduler import FakeClock, FakeNotificationSink, SchedulerService
from services.wake.schemas import WakeEvent
from services.wake.session_manager import SessionManager
from tests.test_core_api import auth, make_container
from tests.test_scheduler import RecordingAudit


class FakeReflection:
    """Records reflect_session calls without touching the runtime."""

    def __init__(self) -> None:
        self.sessions: list[str] = []

    async def reflect_session(self, session_id: str) -> list[object]:
        self.sessions.append(session_id)
        return []


class ExplodingReflection:
    """Fails with content that must never be copied into audit records."""

    def __init__(self) -> None:
        self.sessions: list[str] = []

    async def reflect_session(self, session_id: str) -> list[object]:
        self.sessions.append(session_id)
        raise RuntimeError("transcript-leak-marker")


async def _memory(settings: AprilSettings) -> tuple[Database, SqliteMemory]:
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    return database, SqliteMemory(database)


def test_session_close_api_triggers_reflection(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    reflection = FakeReflection()
    container.archive_reflection = reflection  # type: ignore[assignment]
    client = TestClient(create_app(container))

    created = client.post("/sessions", json={"source": "terminal"}, headers=auth(settings_tmp))
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    closed = client.post(f"/sessions/{session_id}/close", headers=auth(settings_tmp))
    assert closed.status_code == 200
    assert closed.json() == {"session_id": session_id, "closed": True}
    assert reflection.sessions == [session_id]

    # Closing again is idempotent and must not re-run reflection.
    again = client.post(f"/sessions/{session_id}/close", headers=auth(settings_tmp))
    assert again.status_code == 200
    assert again.json()["closed"] is False
    assert reflection.sessions == [session_id]


def test_session_close_api_is_best_effort_when_reflection_fails(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    reflection = ExplodingReflection()
    container.archive_reflection = reflection  # type: ignore[assignment]
    container.session_manager = None
    client = TestClient(create_app(container))

    created = client.post("/sessions", json={"source": "terminal"}, headers=auth(settings_tmp))
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    closed = client.post(f"/sessions/{session_id}/close", headers=auth(settings_tmp))
    assert closed.status_code == 200
    assert closed.json() == {"session_id": session_id, "closed": True}
    assert reflection.sessions == [session_id]

    audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
    assert "archive_reflection_failed" in audit_text
    assert "RuntimeError" in audit_text
    assert "transcript-leak-marker" not in audit_text


def test_session_close_unknown_session_is_404(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post("/sessions/not-a-session/close", headers=auth(settings_tmp))
    assert response.status_code == 404


def test_session_close_requires_auth(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post("/sessions/whatever/close")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_close_idle_sessions_triggers_reflection(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        now = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
        reflection = FakeReflection()
        manager = SessionManager(
            memory,
            continuity_minutes=10.0,
            clock=lambda: now,
            on_close=reflection.reflect_session,
        )
        resolution = await manager.handle_wake(WakeEvent(source="terminal"))

        # Inside the continuity window nothing is closed.
        now = datetime(2026, 7, 3, 12, 5, tzinfo=UTC)
        assert await manager.close_idle_sessions() == []
        assert reflection.sessions == []

        # Past the window the session closes and reflection runs once.
        now = datetime(2026, 7, 3, 12, 30, tzinfo=UTC)
        closed = await manager.close_idle_sessions()
        assert closed == [resolution.session_id]
        assert reflection.sessions == [resolution.session_id]
        record = await memory.get_session(resolution.session_id)
        assert record is not None
        assert record.closed_at is not None

        # Sweep is idempotent.
        assert await manager.close_idle_sessions() == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_stale_session_wake_continues_when_reflection_fails(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        now = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
        reflection = ExplodingReflection()
        manager = SessionManager(
            memory,
            continuity_minutes=10.0,
            clock=lambda: now,
            on_close=reflection.reflect_session,
            audit=AuditLogger(settings_tmp.audit_path),
        )
        first = await manager.handle_wake(WakeEvent(source="terminal"))

        now = datetime(2026, 7, 3, 12, 30, tzinfo=UTC)
        second = await manager.handle_wake(WakeEvent(source="terminal"))

        assert second.session_id != first.session_id
        assert second.joined_existing is False
        assert reflection.sessions == [first.session_id]
        first_record = await memory.get_session(first.session_id)
        second_record = await memory.get_session(second.session_id)
        assert first_record is not None
        assert first_record.closed_at is not None
        assert second_record is not None
        assert second_record.closed_at is None

        entries = [
            json.loads(line)
            for line in settings_tmp.audit_path.read_text(encoding="utf-8").splitlines()
        ]
        failures = [
            entry for entry in entries if entry.get("event_type") == "archive_reflection_failed"
        ]
        assert len(failures) == 1
        assert failures[0]["actor"] == "archive_agent"
        assert failures[0]["reference_id"] == first.session_id
        assert failures[0]["error_type"] == "RuntimeError"
        assert failures[0]["error_message_length"] == len("transcript-leak-marker")
        assert "transcript-leak-marker" not in settings_tmp.audit_path.read_text(
            encoding="utf-8"
        )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_scheduler_tick_closes_idle_sessions(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        start = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
        clock = FakeClock(start)
        reflection = FakeReflection()
        manager = SessionManager(
            memory,
            continuity_minutes=10.0,
            clock=clock.now,
            on_close=reflection.reflect_session,
        )
        resolution = await manager.handle_wake(WakeEvent(source="terminal"))
        audit = RecordingAudit()
        service = SchedulerService(
            settings=settings_tmp,
            memory=memory,
            audit=audit,  # type: ignore[arg-type]
            sink=FakeNotificationSink(),
            clock=clock,
            session_manager=manager,
        )
        await service.tick()
        assert service.closed_idle_session_count == 0

        clock.advance(30 * 60)
        await service.tick()
        assert service.closed_idle_session_count == 1
        assert reflection.sessions == [resolution.session_id]
        assert "scheduler.idle_session_closed" in audit.events()
    finally:
        await database.close()


def test_cli_attach_closes_session_on_quit_and_eof(monkeypatch) -> None:
    import apps.cli.main as cli_main

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def post(self, path: str, payload: object, *, auth: bool = True) -> dict:
            self.calls.append(("POST", path))
            if path == "/sessions":
                return {
                    "session_id": "session-1",
                    "conversation_id": "conversation-1",
                    "joined_existing": False,
                }
            if path == "/sessions/session-1/close":
                return {"session_id": "session-1", "closed": True}
            raise AssertionError(path)

    for exit_input in ("/quit", "/exit", EOFError):
        fake = FakeClient()
        monkeypatch.setattr(cli_main, "client", lambda fake=fake: fake)
        monkeypatch.setattr(cli_main, "_maybe_autostart_daemon", lambda: None)
        if exit_input is EOFError:

            def raise_eof(prompt: str) -> str:
                raise EOFError

            monkeypatch.setattr(cli_main.Prompt, "ask", staticmethod(raise_eof))
        else:
            monkeypatch.setattr(
                cli_main.Prompt, "ask", staticmethod(lambda prompt, value=exit_input: value)
            )
        cli_main.attach()
        assert ("POST", "/sessions/session-1/close") in fake.calls


def test_cli_attach_closes_session_on_interrupt(monkeypatch) -> None:
    import apps.cli.main as cli_main

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def post(self, path: str, payload: object, *, auth: bool = True) -> dict:
            self.calls.append(("POST", path))
            if path == "/sessions":
                return {
                    "session_id": "session-1",
                    "conversation_id": "conversation-1",
                    "joined_existing": True,
                }
            if path == "/sessions/session-1/close":
                return {"session_id": "session-1", "closed": True}
            raise AssertionError(path)

    fake = FakeClient()
    monkeypatch.setattr(cli_main, "client", lambda: fake)
    monkeypatch.setattr(cli_main, "_maybe_autostart_daemon", lambda: None)

    def raise_interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_main.Prompt, "ask", staticmethod(raise_interrupt))
    with pytest.raises(KeyboardInterrupt):
        cli_main.attach()
    assert ("POST", "/sessions/session-1/close") in fake.calls
