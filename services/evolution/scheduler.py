from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

from april_common.settings import AprilSettings
from services.memory.sqlite_memory import SqliteMemory
from services.pool.governor import ResourceGovernor

_LAST_EVOLUTION_DATE_KEY = "last_evolution_date"


@dataclass(frozen=True, slots=True)
class EvolutionGateDecision:
    allowed: bool
    reason: str


class EvolutionSchedulerGate:
    def __init__(
        self,
        settings: AprilSettings,
        memory: SqliteMemory,
        *,
        governor: ResourceGovernor,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.governor = governor

    async def should_run(self, now: datetime) -> EvolutionGateDecision:
        if not self.settings.evolution.enabled:
            return EvolutionGateDecision(False, "evolution disabled")
        if not _inside_window(now.time(), self.settings.evolution.window):
            return EvolutionGateDecision(False, "outside evolution window")
        today = now.date().isoformat()
        if await self.memory.get_scheduler_state(_LAST_EVOLUTION_DATE_KEY) == today:
            return EvolutionGateDecision(False, "already ran today")
        decision = self.governor.assess_background()
        if not decision.allowed:
            return EvolutionGateDecision(False, ",".join(decision.reasons))
        return EvolutionGateDecision(True, "allowed")

    async def mark_ran(self, now: datetime) -> None:
        await self.memory.set_scheduler_state(_LAST_EVOLUTION_DATE_KEY, now.date().isoformat())


def _inside_window(current: time, window: str) -> bool:
    start_raw, _, end_raw = window.partition("-")
    start = _parse_time(start_raw)
    end = _parse_time(end_raw)
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def _parse_time(value: str) -> time:
    hour_raw, _, minute_raw = value.strip().partition(":")
    return time(hour=int(hour_raw), minute=int(minute_raw))
