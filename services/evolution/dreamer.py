from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.consolidate import consolidate_memories
from services.evolution.disarm import disarmed_execution
from services.evolution.evaluator import (
    RuntimeEvalClient,
    evaluate_overlay_candidate,
    evaluate_overlay_candidate_real_runtime,
)
from services.evolution.feedback_eval import count_pending_eval_cases
from services.evolution.playbook_miner import mine_playbook_candidates
from services.evolution.prompt_evolver import OverlayCandidate, generate_overlay_candidates
from services.evolution.replay import collect_replay_samples
from services.evolution.report import phase_status_summary, write_report
from services.evolution.scheduler import EvolutionSchedulerGate
from services.evolution.user_model import update_user_model
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
        clock: Callable[[], float] = time.monotonic,
        nice_applier: Callable[[int], str | None] | None = None,
        runtime_client: RuntimeEvalClient | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.gate = gate
        self.audit = audit
        self.guard = guard or EvolutionWriteGuard(settings, audit=audit)
        self.overlay_manager = overlay_manager or PromptOverlayManager(
            settings, memory.database, audit=audit, guard=self.guard
        )
        self.clock = clock
        # Optional local runtime handle for production real-runtime evals; when
        # absent, production candidates stay pending with an explicit blocker.
        self.runtime_client = runtime_client
        # Renicing is process-wide and irreversible for an unprivileged process,
        # so it is only applied when a dedicated dreamer worker injects an
        # applier (see run_standalone). The in-process scheduler path must never
        # deprioritize the shared Core API process.
        self.nice_applier = nice_applier

    async def run_once(self, now: datetime) -> DreamerRunResult:
        decision = await self.gate.should_run(now)
        if not decision.allowed:
            return DreamerRunResult("skipped", decision.reason)
        self._apply_nice()
        run_id = str(uuid.uuid4())
        started = self.clock()
        budget_seconds = float(self.settings.evolution.max_minutes) * 60.0
        phases: dict[str, dict[str, Any]] = {}

        def budget_left() -> bool:
            return (self.clock() - started) < budget_seconds

        candidates: list[OverlayCandidate] | None = []
        work_phases: list[tuple[str, Callable[[], Any]]] = [
            ("replay", lambda: self._phase_replay(now)),
            ("distill", self._phase_distill),
            ("mine", self._phase_mine),
            ("evolve", self._phase_evolve),
            (
                "examine",
                lambda: self._phase_examine(candidates or [], run_id=run_id),
            ),
        ]
        for name, phase_factory in work_phases:
            if not budget_left():
                # The wall-clock budget is enforced between phases: a phase in
                # flight finishes, later phases are skipped and say why.
                phases[name] = {
                    "status": "skipped",
                    "reason": "wall clock budget exhausted "
                    f"(evolution.max_minutes={self.settings.evolution.max_minutes})",
                }
                self._audit("dreamer_phase_skipped_budget", run_id=run_id, detail=name)
                continue
            result = await self._run_phase(phases, name, phase_factory())
            if name == "evolve":
                candidates = result

        report_path, report = write_report(
            self.settings,
            guard=self.guard,
            run_id=run_id,
            phases=phases,
            pending_eval_cases=count_pending_eval_cases(self.settings),
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

    async def _run_phase(self, phases: dict[str, dict[str, Any]], name: str, coroutine: Any) -> Any:
        """Run one phase in isolation: a failure is recorded, never propagated.

        Every phase runs disarmed: normal Level >= 1 tool execution routed
        through the permission layer is refused for the duration of the phase.
        """
        try:
            with disarmed_execution(name):
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
        # Deterministic decay/fading of stale machine memories runs with the
        # same fence; nothing is ever deleted, fading rows stay inspectable.
        from services.memory.decay import apply_memory_decay

        decay = await apply_memory_decay(self.memory, guard=self.guard)
        user_model = await update_user_model(
            self.memory,
            self.settings,
            guard=self.guard,
        )
        return report, {
            **report.to_payload(),
            **decay.to_payload(),
            "user_model": user_model.to_payload(),
        }

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
            # Honest labelling: overlay candidates come from deterministic
            # templates over local feedback, not from an LLM. Any improvement
            # claim must come from the examine-phase evals, never from here.
            "method": "deterministic-heuristic",
        }
        return candidates, payload

    async def _phase_examine(
        self, candidates: list[OverlayCandidate], *, run_id: str
    ) -> tuple[Any, dict[str, Any]]:
        activated: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        approval_required: list[dict[str, Any]] = []
        pending_real_runtime: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        blockers: list[str] = []
        require_real_runtime = self.settings.environment == "production"
        real_runtime_counts = {"passed": 0, "failed": 0, "skipped": 0}
        for candidate in candidates:
            evaluation = evaluate_overlay_candidate(
                agent=candidate.agent, content=candidate.content, settings=self.settings
            )
            evaluation_payload = evaluation.to_payload()
            active_score = await self.overlay_manager.active_eval_score(candidate.agent)
            apply_baseline = max(
                evaluation.baseline,
                active_score if active_score is not None else evaluation.baseline,
            )
            if evaluation.score < apply_baseline:
                evaluations.append(evaluation_payload)
                reason = "below current baseline" if active_score is not None else "below baseline"
                discarded.append({"agent": candidate.agent, "reason": reason})
                continue
            if require_real_runtime:
                # Production never activates an overlay on deterministic/fake
                # replay alone: a real local-runtime eval must pass, otherwise
                # the candidate stays pending with an explicit reason.
                real_eval = await evaluate_overlay_candidate_real_runtime(
                    agent=candidate.agent,
                    content=candidate.content,
                    settings=self.settings,
                    runtime_client=self.runtime_client,
                )
                evaluation_payload["real_runtime"] = real_eval.to_payload()
                evaluations.append(evaluation_payload)
                if real_eval.passed:
                    real_runtime_counts["passed"] += 1
                elif real_eval.skipped:
                    real_runtime_counts["skipped"] += 1
                else:
                    real_runtime_counts["failed"] += 1
                if not real_eval.passed:
                    blockers.extend(real_eval.blockers)
                    pending_real_runtime.append(
                        {
                            "agent": candidate.agent,
                            "status": real_eval.status,
                            "reason": (
                                "; ".join(real_eval.blockers)
                                if real_eval.blockers
                                else "real-runtime evaluation did not pass"
                            ),
                        }
                    )
                    self._audit(
                        "dreamer_overlay_pending_real_runtime",
                        run_id=run_id,
                        detail=f"{candidate.agent}: {real_eval.status}",
                    )
                    continue
            else:
                evaluations.append(evaluation_payload)
            result = await self.overlay_manager.apply_candidate(
                agent=candidate.agent,
                content=candidate.content,
                eval_score=evaluation.score,
                baseline_score=apply_baseline,
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
            "pending_real_runtime": pending_real_runtime,
            "eval_modes": {
                "environment": self.settings.environment,
                "real_runtime_required": require_real_runtime,
                "deterministic_fixture_passed": sum(
                    1 for item in evaluations if item.get("deterministic_fixture_passed") is True
                ),
                "real_runtime_eval_passed": real_runtime_counts["passed"],
                "real_runtime_eval_failed": real_runtime_counts["failed"],
                "real_runtime_eval_skipped": real_runtime_counts["skipped"],
                "blockers": sorted(set(blockers)),
            },
        }
        return payload, payload

    def _apply_nice(self) -> None:
        """Lower this process's priority when a dedicated worker asked for it."""
        if self.nice_applier is None:
            return
        error = self.nice_applier(self.settings.governor.dreamer_nice)
        if error is None:
            self._audit(
                "dreamer_nice_applied",
                run_id=None,
                detail=f"nice={self.settings.governor.dreamer_nice}",
            )
        else:
            # Failing to renice is never fatal; the run continues at normal
            # priority with the reason on record.
            self._audit("dreamer_nice_failed", run_id=None, detail=error)

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


def process_nice_applier(nice: int) -> str | None:
    """Renice the *current* process. Only safe in a dedicated dreamer worker.

    Unprivileged processes cannot lower niceness back down afterwards, which is
    exactly why the in-process scheduler path never uses this.
    """
    import os

    try:
        os.setpriority(os.PRIO_PROCESS, 0, nice)
    except (AttributeError, OSError, PermissionError) as exc:
        return f"setpriority failed: {exc}"
    return None


def _report_sort_key(path: Any) -> tuple[str, str]:
    """Newest-first key: embedded created_at wins, filesystem mtime breaks ties.

    Report filenames are UUIDs, so lexicographic filename order says nothing
    about recency; the report's own created_at field is authoritative.
    """
    created_at = ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            created_at = str(payload.get("created_at", ""))
    except (OSError, json.JSONDecodeError):
        created_at = ""
    try:
        mtime = f"{path.stat().st_mtime:020.6f}"
    except OSError:
        mtime = ""
    return (created_at, mtime)


def latest_report(settings: AprilSettings) -> dict[str, Any] | None:
    reports = list((settings.evolution_path / "reports").glob("*.json"))
    if not reports:
        return None
    newest = max(reports, key=_report_sort_key)
    try:
        return json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


async def run_standalone(settings: AprilSettings, now: datetime) -> DreamerRunResult:
    """Dedicated dreamer worker: own DB connection, governor gate, and renice.

    This is the only path that applies governor.dreamer_nice — it deprioritizes
    a process whose whole job is the nightly dream, never the shared Core API.
    """
    from april_common.audit import AuditLogger
    from services.april_runtime.client import RuntimeClient
    from services.memory.database import Database
    from services.memory.migrations import run_migrations
    from services.pool.governor import ResourceGovernor

    database = Database(settings.database_path)
    await database.connect()
    try:
        await run_migrations(database)
        memory = SqliteMemory(database)
        audit = AuditLogger(settings.audit_path)
        gate = EvolutionSchedulerGate(settings, memory, governor=ResourceGovernor(settings))
        dreamer = DreamerService(
            settings,
            memory=memory,
            gate=gate,
            audit=audit,
            nice_applier=process_nice_applier,
            # Loopback-only local runtime; used solely for real-runtime evals.
            runtime_client=RuntimeClient(
                settings.runtime.url,
                timeout=settings.runtime.request_timeout_seconds,
                token=settings.runtime.token,
            ),
        )
        return await dreamer.run_once(now)
    finally:
        await database.close()


def main() -> None:
    """`python -m services.evolution.dreamer`: run one gated dream cycle."""
    import asyncio

    from april_common.settings import get_settings
    from april_common.time import utc_now

    result = asyncio.run(run_standalone(get_settings(), utc_now()))
    print(json.dumps({"status": result.status, "reason": result.reason}))


if __name__ == "__main__":
    main()
