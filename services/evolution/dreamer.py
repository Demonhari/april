from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.consolidate import consolidate_memories
from services.evolution.evaluator import evaluate_overlay_candidate
from services.evolution.playbook_miner import mine_playbook_candidates
from services.evolution.prompt_evolver import OverlayCandidate, generate_overlay_candidates
from services.evolution.replay import collect_replay_samples
from services.evolution.report import phase_status_summary, write_report
from services.evolution.scheduler import EvolutionSchedulerGate
from services.evolution.versions import PromptOverlayManager
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.sqlite_memory import SqliteMemory

DREAMER_PHASES = ("replay", "distill", "mine", "evolve", "examine", "report")


@dataclass(frozen=True, slots=True)
class DreamerRunResult:
    status: Literal["skipped", "completed"]
    reason: str
    report_path: str | None = None


class DreamerService:
    """Nightly self-evolution: a deterministic, auditable phase runner.

    Six phases run in order — D1 replay, D2 distill/consolidate, D3 mine,
    D4 evolve, D5 examine, D6 rest/report. Every phase is fenced by the
    evolution write guard (data/evolution, data/playbooks, approved tables
    only), fails independently without aborting the run, and leaves its
    outcome in the report and audit log. Deleting data/evolution restores
    stock behaviour.
    """

    def __init__(
        self,
        settings: AprilSettings,
        *,
        memory: SqliteMemory,
        gate: EvolutionSchedulerGate,
        audit: AuditLogger | None = None,
        guard: EvolutionWriteGuard | None = None,
        overlay_manager: PromptOverlayManager | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.gate = gate
        self.audit = audit
        self.guard = guard or EvolutionWriteGuard(settings, audit=audit)
        self.overlay_manager = overlay_manager or PromptOverlayManager(
            settings, memory.database, audit=audit, guard=self.guard
        )

    async def run_once(self, now: datetime) -> DreamerRunResult:
        decision = await self.gate.should_run(now)
        if not decision.allowed:
            return DreamerRunResult("skipped", decision.reason)
        run_id = str(uuid.uuid4())
        phases: dict[str, dict[str, Any]] = {}

        await self._run_phase(phases, "replay", self._phase_replay(now))
        await self._run_phase(phases, "distill", self._phase_distill())
        await self._run_phase(phases, "mine", self._phase_mine())
        candidates = await self._run_phase(phases, "evolve", self._phase_evolve())
        await self._run_phase(
            phases, "examine", self._phase_examine(candidates or [], run_id=run_id)
        )

        report_path, report = write_report(
            self.settings, guard=self.guard, run_id=run_id, phases=phases
        )
        phases["report"] = {"status": "completed", "path": str(report_path)}
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
                json.dumps(phase_status_summary(phases), sort_keys=True),
                str(report_path),
                report["created_at"],
                utc_now_iso(),
            ),
        )
        await self.gate.mark_ran(now)
        self._audit(
            "dreamer_run_completed",
            run_id=run_id,
            detail=str(report.get("summary", "")),
        )
        return DreamerRunResult("completed", "completed", report_path=str(report_path))

    async def _run_phase(
        self, phases: dict[str, dict[str, Any]], name: str, coroutine: Any
    ) -> Any:
        """Run one phase in isolation: a failure is recorded, never propagated."""
        try:
            result, payload = await coroutine
        except Exception as exc:
            phases[name] = {"status": "failed", "error": str(exc)[:500]}
            self._audit("dreamer_phase_failed", run_id=None, detail=f"{name}: {exc}")
            return None
        phases[name] = {"status": "completed", **payload}
        return result

    async def _phase_replay(self, now: datetime) -> tuple[Any, dict[str, Any]]:
        seed = int(now.strftime("%Y%m%d"))
        report = await collect_replay_samples(self.memory, seed=seed)
        return report, report.to_payload()

    async def _phase_distill(self) -> tuple[Any, dict[str, Any]]:
        report = await consolidate_memories(self.memory, guard=self.guard)
        return report, report.to_payload()

    async def _phase_mine(self) -> tuple[Any, dict[str, Any]]:
        report = await mine_playbook_candidates(self.memory, self.settings, guard=self.guard)
        return report, report.to_payload()

    async def _phase_evolve(self) -> tuple[list[OverlayCandidate], dict[str, Any]]:
        candidates = await generate_overlay_candidates(self.memory, self.settings)
        # Candidates are data only. They are persisted for review even when the
        # examine phase later discards them or holds them for approval.
        stored: list[str] = []
        for index, candidate in enumerate(candidates):
            path = (
                self.settings.evolution_path
                / "candidates"
                / f"{candidate.agent}-{index}.overlay.txt"
            )
            stored.append(str(self.guard.write_text(path, candidate.content)))
        payload = {
            "candidates": [candidate.to_payload() for candidate in candidates],
            "stored_paths": stored,
        }
        return candidates, payload

    async def _phase_examine(
        self, candidates: list[OverlayCandidate], *, run_id: str
    ) -> tuple[Any, dict[str, Any]]:
        activated: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        approval_required: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        for candidate in candidates:
            evaluation = evaluate_overlay_candidate(
                agent=candidate.agent, content=candidate.content, settings=self.settings
            )
            evaluations.append(evaluation.to_payload())
            if not evaluation.passed:
                discarded.append({"agent": candidate.agent, "reason": "below baseline"})
                continue
            result = await self.overlay_manager.apply_candidate(
                agent=candidate.agent,
                content=candidate.content,
                eval_score=evaluation.score,
                baseline_score=evaluation.baseline,
                source="dreamer",
            )
            entry = {"agent": candidate.agent, "version": result.version}
            if result.status == "applied":
                activated.append(entry)
            elif result.status == "approval_required":
                approval_required.append({**entry, "reason": result.reason})
            else:
                discarded.append({**entry, "reason": result.reason})
        payload = {
            "evaluations": evaluations,
            "activated": activated,
            "discarded": discarded,
            "approval_required": approval_required,
        }
        return payload, payload

    def _audit(self, event_type: str, *, run_id: str | None, detail: str | None = None) -> None:
        if self.audit is not None:
            self.audit.write(
                {
                    "event_type": event_type,
                    "actor": "dreamer",
                    "run_id": run_id,
                    "detail": detail,
                }
            )


def latest_report(settings: AprilSettings) -> dict[str, Any] | None:
    reports = sorted((settings.evolution_path / "reports").glob("*.json"))
    if not reports:
        return None
    try:
        return json.loads(reports[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
