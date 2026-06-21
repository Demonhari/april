from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.brain.planner import TaskPlan, TaskStep
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.scheduler import (
    FakeClock,
    FakeNotificationSink,
    SchedulerService,
    compose_briefing,
)
from services.scheduler.notifications import Notification, NotificationSink


class RecordingAudit:
    """Captures audit entries in memory so tests can assert on emitted events."""

    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def write(self, entry: dict[str, object]) -> None:
        self.entries.append(entry)

    def events(self) -> list[object]:
        return [entry.get("event") for entry in self.entries]


class RaiseOnKindSink(NotificationSink):
    """Raises when emitting a chosen notification kind; records the rest."""

    def __init__(self, fail_kind: str) -> None:
        self.fail_kind = fail_kind
        self.emitted: list[Notification] = []

    async def emit(self, notification: Notification) -> None:
        if notification.kind == self.fail_kind:
            raise RuntimeError("sink boom")
        self.emitted.append(notification)


async def _memory(settings: AprilSettings) -> tuple[Database, SqliteMemory]:
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    return database, SqliteMemory(database)


def _scheduler_settings(base: AprilSettings, **overrides: object) -> AprilSettings:
    return base.model_copy(update={"scheduler": base.scheduler.model_copy(update=overrides)})


def _task(*, conversation_id: str, intent: str, title: str, status: str) -> TaskPlan:
    return TaskPlan(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        request_id="request",
        intent=intent,
        agent="general_agent",
        model_id="april-brain",
        steps=[TaskStep(index=1, title=title)],
        status=status,
        created_at=utc_now_iso(),
    )


def _briefings(sink: FakeNotificationSink) -> list[Notification]:
    return [item for item in sink.emitted if item.kind == "briefing"]


async def test_reminder_fires_exactly_once(settings_tmp: AprilSettings) -> None:
    settings = _scheduler_settings(settings_tmp, enabled=True)
    database, memory = await _memory(settings)
    await memory.create_reminder("Drink water", due_at="2026-06-21T09:00:00Z")
    clock = FakeClock(datetime(2026, 6, 21, 12, 0, tzinfo=UTC))
    sink = FakeNotificationSink()
    service = SchedulerService(
        settings=settings,
        memory=memory,
        audit=RecordingAudit(),  # type: ignore[arg-type]
        sink=sink,
        clock=clock,
    )

    await service.tick()
    assert len(sink.emitted) == 1
    assert sink.emitted[0].kind == "reminder"
    assert sink.emitted[0].body == "Drink water"
    assert service.fired_reminder_count == 1

    await service.tick()
    assert len(sink.emitted) == 1
    assert service.fired_reminder_count == 1
    await database.close()


async def test_scheduler_off_by_default(settings_tmp: AprilSettings) -> None:
    database, memory = await _memory(settings_tmp)
    assert settings_tmp.scheduler.enabled is False
    service = SchedulerService(
        settings=settings_tmp,
        memory=memory,
        audit=RecordingAudit(),  # type: ignore[arg-type]
        sink=FakeNotificationSink(),
        clock=FakeClock(datetime(2026, 6, 21, 12, 0, tzinfo=UTC)),
    )
    await service.start()
    assert service.running is False
    await service.stop()
    await database.close()


async def test_briefing_fires_once_per_local_day(settings_tmp: AprilSettings) -> None:
    settings = _scheduler_settings(
        settings_tmp, enabled=True, briefing_enabled=True, briefing_time="08:00"
    )
    database, memory = await _memory(settings)
    clock = FakeClock(datetime(2026, 6, 21, 7, 59, tzinfo=UTC))
    sink = FakeNotificationSink()
    service = SchedulerService(
        settings=settings,
        memory=memory,
        audit=RecordingAudit(),  # type: ignore[arg-type]
        sink=sink,
        clock=clock,
        local_tz=UTC,
    )

    await service.tick()
    assert _briefings(sink) == []
    assert service.fired_briefing_count == 0

    clock.set(datetime(2026, 6, 21, 8, 0, tzinfo=UTC))
    await service.tick()
    assert len(_briefings(sink)) == 1
    assert service.fired_briefing_count == 1

    await service.tick()
    assert len(_briefings(sink)) == 1
    assert service.fired_briefing_count == 1

    clock.set(datetime(2026, 6, 22, 8, 0, tzinfo=UTC))
    await service.tick()
    assert len(_briefings(sink)) == 2
    assert service.fired_briefing_count == 2
    await database.close()


async def test_briefing_persists_across_restart(settings_tmp: AprilSettings) -> None:
    settings = _scheduler_settings(
        settings_tmp, enabled=True, briefing_enabled=True, briefing_time="08:00"
    )
    database, memory = await _memory(settings)

    first_sink = FakeNotificationSink()
    first = SchedulerService(
        settings=settings,
        memory=memory,
        audit=RecordingAudit(),  # type: ignore[arg-type]
        sink=first_sink,
        clock=FakeClock(datetime(2026, 6, 21, 8, 0, tzinfo=UTC)),
        local_tz=UTC,
    )
    await first.tick()
    assert len(_briefings(first_sink)) == 1

    # Simulate a process restart: brand new service on the same persisted memory.
    second_sink = FakeNotificationSink()
    second = SchedulerService(
        settings=settings,
        memory=memory,
        audit=RecordingAudit(),  # type: ignore[arg-type]
        sink=second_sink,
        clock=FakeClock(datetime(2026, 6, 21, 9, 0, tzinfo=UTC)),
        local_tz=UTC,
    )
    await second.tick()
    assert _briefings(second_sink) == []
    assert second.fired_briefing_count == 0
    await database.close()


async def test_briefing_composition_filters(settings_tmp: AprilSettings) -> None:
    database, memory = await _memory(settings_tmp)
    conversation_id = await memory.create_conversation()
    await memory.create_task_plan(
        _task(
            conversation_id=conversation_id,
            intent="report",
            title="Write report",
            status="planned",
        )
    )
    await memory.create_task_plan(
        _task(conversation_id=conversation_id, intent="bug", title="Fix bug", status="running")
    )
    await memory.create_task_plan(
        _task(
            conversation_id=conversation_id,
            intent="archive",
            title="Archived task",
            status="completed",
        )
    )
    now = datetime(2026, 6, 21, 8, 0, tzinfo=UTC)
    in_window = (now + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    out_of_window = (now + timedelta(hours=48)).isoformat().replace("+00:00", "Z")
    await memory.create_reminder("Standup", due_at=in_window)
    await memory.create_reminder("Quarterly review", due_at=out_of_window)

    now_iso = now.isoformat().replace("+00:00", "Z")
    until_iso = (now + timedelta(hours=24)).isoformat().replace("+00:00", "Z")
    notification = await compose_briefing(memory, now_iso=now_iso, until_iso=until_iso)

    assert notification.kind == "briefing"
    body = notification.body
    assert "Write report" in body
    assert "Fix bug" in body
    assert "Archived task" not in body
    assert "Standup" in body
    assert "Quarterly review" not in body
    await database.close()


async def test_reminder_and_briefing_paths_are_independent(
    settings_tmp: AprilSettings,
) -> None:
    settings = _scheduler_settings(
        settings_tmp, enabled=True, briefing_enabled=True, briefing_time="08:00"
    )

    # Reminder emit fails -> briefing path still runs.
    database, memory = await _memory(settings)
    await memory.create_reminder("Drink water", due_at="2026-06-21T07:00:00Z")
    audit = RecordingAudit()
    sink = RaiseOnKindSink("reminder")
    service = SchedulerService(
        settings=settings,
        memory=memory,
        audit=audit,  # type: ignore[arg-type]
        sink=sink,
        clock=FakeClock(datetime(2026, 6, 21, 8, 0, tzinfo=UTC)),
        local_tz=UTC,
    )
    await service.tick()
    assert "scheduler.reminder_error" in audit.events()
    assert [item.kind for item in sink.emitted] == ["briefing"]
    assert service.fired_briefing_count == 1
    await database.close()

    # Vice versa: briefing emit fails -> reminder path still runs. A new day so the
    # persisted last_briefing_date from the first scenario does not suppress the briefing.
    database, memory = await _memory(settings)
    await memory.create_reminder("Drink water", due_at="2026-06-22T07:00:00Z")
    audit = RecordingAudit()
    sink = RaiseOnKindSink("briefing")
    service = SchedulerService(
        settings=settings,
        memory=memory,
        audit=audit,  # type: ignore[arg-type]
        sink=sink,
        clock=FakeClock(datetime(2026, 6, 22, 8, 0, tzinfo=UTC)),
        local_tz=UTC,
    )
    await service.tick()
    assert "scheduler.briefing_error" in audit.events()
    assert [item.kind for item in sink.emitted] == ["reminder"]
    assert service.fired_reminder_count == 1
    await database.close()
