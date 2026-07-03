from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.scheduler import EvolutionSchedulerGate
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.sqlite_memory import SqliteMemory

DREAMER_PHASES = ("replay", "distill", "mine", "evolve", "examine", "report")


@dataclass(frozen=True, slots=True)
class DreamerRunResult:
    status: Literal["skipped", "completed"]
    reason: str
    report_path: str | None = None


class DreamerService:
    def __init__(
        self,
        settings: AprilSettings,
        *,
        memory: SqliteMemory,
        gate: EvolutionSchedulerGate,
        audit: AuditLogger | None = None,
        guard: EvolutionWriteGuard | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.gate = gate
        self.audit = audit
        self.guard = guard or EvolutionWriteGuard(settings, audit=audit)

    async def run_once(self, now: datetime) -> DreamerRunResult:
        decision = await self.gate.should_run(now)
        if not decision.allowed:
            return DreamerRunResult("skipped", decision.reason)
        run_id = str(uuid.uuid4())
        phases = dict.fromkeys(DREAMER_PHASES, "no_candidates")
        report = {
            "run_id": run_id,
            "status": "completed",
            "reason": "no evolution candidates were produced",
            "phases": phases,
            "created_at": utc_now_iso(),
        }
        report_path = self.settings.evolution_path / "reports" / f"{run_id}.json"
        written = self.guard.write_text(report_path, json.dumps(report, sort_keys=True, indent=2))
        self.guard.validate_table("evolution_runs")
        await self.memory.database.execute(
            """
            INSERT INTO evolution_runs(
                id, date, status, phases_json, report_path, created_at, completed_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now.date().isoformat(),
                "completed",
                json.dumps(phases, sort_keys=True),
                str(written),
                report["created_at"],
                utc_now_iso(),
            ),
        )
        await self.gate.mark_ran(now)
        if self.audit is not None:
            self.audit.write({"event_type": "dreamer_run_completed", "run_id": run_id})
        return DreamerRunResult("completed", "completed", report_path=str(written))


def latest_report(settings: AprilSettings) -> dict[str, Any] | None:
    reports = sorted((settings.evolution_path / "reports").glob("*.json"))
    if not reports:
        return None
    try:
        return json.loads(reports[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
