from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from april_common.time import utc_now


class Clock:
    """Injectable time source so the scheduler loop never has to really sleep in tests."""

    def now(self) -> datetime:
        raise NotImplementedError

    async def sleep(self, seconds: float) -> None:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> datetime:
        return utc_now()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeClock(Clock):
    """Deterministic clock for tests: time only moves when the test advances it,
    and sleep returns immediately while recording the requested durations."""

    def __init__(self, start: datetime) -> None:
        self._now = start
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)

    def set(self, value: datetime) -> None:
        self._now = value
