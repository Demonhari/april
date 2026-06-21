from __future__ import annotations

import asyncio
from datetime import UTC

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.memory.sqlite_memory import SqliteMemory
from services.scheduler.clock import Clock, SystemClock
from services.scheduler.notifications import Notification, NotificationSink


class SchedulerService:
    """Pure-asyncio poll loop that fires due reminders through a notification sink.

    OFF by default: start() is a no-op unless scheduler.enabled is true. The loop is
    driven by an injectable Clock so tests advance time and call tick() without sleeping.
    """

    def __init__(
        self,
        *,
        settings: AprilSettings,
        memory: SqliteMemory,
        audit: AuditLogger,
        sink: NotificationSink,
        clock: Clock | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.audit = audit
        self.sink = sink
        self.clock = clock or SystemClock()
        self._task: asyncio.Task[None] | None = None
        self._fired_reminders = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def fired_reminder_count(self) -> int:
        return self._fired_reminders

    async def start(self) -> None:
        if not self.settings.scheduler.enabled or self.running:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        interval = max(1.0, float(self.settings.scheduler.poll_interval_seconds))
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let the background loop die
                self.audit.write({"event": "scheduler.error", "error": str(exc)})
            await self.clock.sleep(interval)

    async def tick(self) -> None:
        """One poll pass. Safe to call directly from tests with a FakeClock."""
        await self._fire_due_reminders()

    async def _fire_due_reminders(self) -> None:
        now_iso = self._now_iso()
        for reminder in await self.memory.list_due_reminders(now_iso):
            if not await self.memory.mark_reminder_fired(reminder.id, now_iso):
                # Lost the race to another tick; it has already fired exactly once.
                continue
            self.audit.write(
                {
                    "event": "scheduler.reminder_fired",
                    "reminder_id": reminder.id,
                    "content": reminder.content,
                    "due_at": reminder.due_at,
                    "fired_at": now_iso,
                }
            )
            await self.sink.emit(
                Notification(
                    kind="reminder",
                    title="APRIL Reminder",
                    body=reminder.content,
                    reference_id=reminder.id,
                    created_at=now_iso,
                )
            )
            self._fired_reminders += 1

    def _now_iso(self) -> str:
        return self.clock.now().astimezone(UTC).isoformat().replace("+00:00", "Z")
