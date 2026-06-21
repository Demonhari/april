from __future__ import annotations

import asyncio
from datetime import UTC, datetime, time, timedelta, tzinfo

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.memory.sqlite_memory import SqliteMemory
from services.scheduler.briefing import compose_briefing
from services.scheduler.clock import Clock, SystemClock
from services.scheduler.notifications import Notification, NotificationSink
from services.scheduler.repo_monitor import RepoActivity, compute_repo_activity

_LAST_BRIEFING_KEY = "last_briefing_date"


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
        local_tz: tzinfo | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.audit = audit
        self.sink = sink
        self.clock = clock or SystemClock()
        self._local_tz_override = local_tz
        self._task: asyncio.Task[None] | None = None
        self._fired_reminders = 0
        self._fired_briefings = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def fired_reminder_count(self) -> int:
        return self._fired_reminders

    @property
    def fired_briefing_count(self) -> int:
        return self._fired_briefings

    def _local_tz(self) -> tzinfo:
        if self._local_tz_override is not None:
            return self._local_tz_override
        # Fall back to the real Mac's system local timezone in production.
        local = datetime.now().astimezone().tzinfo
        return local or UTC

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
        """One poll pass. Safe to call directly from tests with a FakeClock.

        Reminders and briefings are processed independently: if one path raises it is
        audited and the other still runs. The outer _run() loop guards the whole tick.
        """
        try:
            await self._fire_due_reminders()
        except Exception as exc:
            self.audit.write({"event": "scheduler.reminder_error", "error": str(exc)})
        try:
            await self._maybe_fire_briefing()
        except Exception as exc:
            self.audit.write({"event": "scheduler.briefing_error", "error": str(exc)})

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

    async def _maybe_fire_briefing(self) -> None:
        if not self.settings.scheduler.briefing_enabled:
            return
        local_now = self.clock.now().astimezone(self._local_tz())
        today_str = local_now.date().isoformat()
        last = await self.memory.get_scheduler_state(_LAST_BRIEFING_KEY)
        if last == today_str:
            return
        if local_now.time() < self._briefing_time():
            return
        now = self.clock.now()
        repo_activity: list[RepoActivity] | None = None
        if self.settings.scheduler.repo_monitor_enabled:
            # Scheduled briefing advances the per-project baseline (persist=True).
            repo_activity = await compute_repo_activity(self.memory, persist=True)
        notification = await compose_briefing(
            self.memory,
            now_iso=self._iso(now),
            until_iso=self._iso(now + timedelta(hours=24)),
            repo_activity=repo_activity,
        )
        await self.sink.emit(notification)
        await self.memory.set_scheduler_state(_LAST_BRIEFING_KEY, today_str)
        self.audit.write({"event": "scheduler.briefing_fired", "date": today_str})
        self._fired_briefings += 1

    def _briefing_time(self) -> time:
        hours, _, minutes = self.settings.scheduler.briefing_time.partition(":")
        return time(hour=int(hours), minute=int(minutes))

    def _now_iso(self) -> str:
        return self._iso(self.clock.now())

    @staticmethod
    def _iso(value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
