from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

import aiosqlite

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from april_common.time import parse_utc_iso, utc_now, utc_now_iso
from services.evolution.eval_review import list_reviewed_eval_cases
from services.evolution.evaluator import (
    RuntimeEvalClient,
    evaluate_overlay_candidate_real_runtime,
)
from services.evolution.versions import prompt_overlay_rejection_reason
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database
from services.permissions.approvals import ApprovalStore, canonical_hash
from services.permissions.schemas import ApprovalRequest

CandidateType = Literal["prompt_overlay", "lora_adapter"]
RolloutState = Literal[
    "candidate",
    "shadow_pending",
    "shadow_running",
    "shadow_passed",
    "canary_pending_approval",
    "canary_running",
    "canary_passed",
    "activation_pending_approval",
    "active",
    "failed",
    "cancelled",
    "rolled_back",
    "rejected",
]

TERMINAL_STATES = frozenset({"failed", "cancelled", "rolled_back", "rejected"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
_SAFE_OUTCOME_KEYS = frozenset(
    {
        "structured_output_valid",
        "repair_attempted",
        "tool_success",
        "tool_failure",
        "approval_denied",
        "user_correction",
        "negative_feedback",
        "regeneration",
        "coding_test_passed",
        "coding_test_failed",
        "runtime_failure",
        "candidate_fallback",
        "hard_failure",
        "latency_ms",
        "baseline_latency_ms",
        "success",
    }
)

_TRANSITIONS: dict[str, frozenset[str]] = {
    "candidate": frozenset({"shadow_pending", "cancelled", "rejected", "failed"}),
    "shadow_pending": frozenset({"shadow_running", "cancelled", "failed"}),
    "shadow_running": frozenset({"shadow_passed", "cancelled", "failed"}),
    "shadow_passed": frozenset(
        {"canary_pending_approval", "cancelled", "rejected", "failed"}
    ),
    "canary_pending_approval": frozenset(
        {"canary_running", "cancelled", "rejected", "failed"}
    ),
    "canary_running": frozenset(
        {"canary_passed", "cancelled", "rolled_back", "failed"}
    ),
    "canary_passed": frozenset(
        {"activation_pending_approval", "cancelled", "rolled_back", "failed"}
    ),
    "activation_pending_approval": frozenset(
        {"active", "cancelled", "rolled_back", "failed"}
    ),
    "active": frozenset({"rolled_back", "failed"}),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "rolled_back": frozenset(),
    "rejected": frozenset(),
}


class RolloutError(RuntimeError):
    """A safe, operator-actionable rollout failure code."""


class InvalidRolloutTransition(RolloutError):
    pass


class RolloutBlocked(RolloutError):
    pass


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    id: str
    candidate_type: CandidateType
    target_id: str
    candidate_id: str
    candidate_sha256: str
    candidate_artifact_path: str
    baseline_id: str
    baseline_sha256: str
    baseline_artifact_path: str | None
    state: RolloutState
    configuration_sha256: str
    shadow_dataset_sha256: str | None
    shadow_evidence_sha256: str | None
    requested_minimum_samples: int
    completed_sample_count: int
    canary_traffic_fraction: float
    canary_max_eligible_turns: int | None
    canary_eligible_turn_count: int
    canary_selected_turn_count: int
    canary_expires_at: str | None
    metrics: dict[str, Any]
    reason_code: str | None
    canary_approval_id: str | None
    activation_approval_id: str | None
    previous_active_artifact: dict[str, Any] | None
    transition_phase: str | None
    shadow_job_id: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    rolled_back_at: str | None
    version: int

    def to_safe_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("candidate_artifact_path", None)
        payload.pop("baseline_artifact_path", None)
        previous = payload.get("previous_active_artifact")
        if isinstance(previous, dict):
            payload["previous_active_artifact"] = {
                key: value
                for key, value in previous.items()
                if key in {"id", "version", "sha256"}
            }
        return payload


@dataclass(frozen=True, slots=True)
class ShadowMetrics:
    sample_count: int
    human_reviewed_sample_count: int
    baseline_pass_count: int
    candidate_pass_count: int
    baseline_structured_valid_count: int
    candidate_structured_valid_count: int
    tool_selection_sample_count: int = 0
    baseline_tool_selection_correct_count: int = 0
    candidate_tool_selection_correct_count: int = 0
    coding_test_sample_count: int = 0
    baseline_coding_test_pass_count: int = 0
    candidate_coding_test_pass_count: int = 0
    baseline_failure_count: int = 0
    candidate_failure_count: int = 0
    baseline_latency_ms: float = 0.0
    candidate_latency_ms: float = 0.0
    baseline_compared: bool = True
    human_reviewed_evidence_present: bool = True
    training_metric_only: bool = False
    hard_failure: bool = False

    def safe_payload(self) -> dict[str, int | float | bool]:
        return asdict(self)


class ShadowEvaluator(Protocol):
    async def evaluate(
        self,
        rollout: RolloutRecord,
        *,
        cancellation_event: asyncio.Event | None = None,
    ) -> ShadowMetrics: ...


@dataclass(frozen=True, slots=True)
class CanaryContext:
    stable_request_id: str
    source: str = "chat"
    mode: str = "standard"
    permission_level: int = 1
    risk_level: str = "none"
    agent: str = "general_agent"
    tool_names: tuple[str, ...] = ()
    has_pending_approval: bool = False
    destructive: bool = False
    external_side_effect: bool = False
    security_sensitive: bool = False
    database_write: bool = False
    repository_write: bool = False
    live_voice: bool = False
    background_evolution: bool = False
    high_risk_reasoning: bool = False


@dataclass(frozen=True, slots=True)
class CanarySelection:
    rollout_id: str | None
    selected: bool
    eligible: bool
    reason_code: str
    overlay_text: str | None = None


@dataclass(frozen=True, slots=True)
class PromotionReadiness:
    runtime_healthy: bool
    database_healthy: bool


FaultHook = Callable[[str, RolloutRecord], Awaitable[None] | None]


class RealPromptShadowEvaluator:
    """A/B evaluation through the existing reviewed-case evaluator.

    It deliberately exposes only aggregates. The evaluator receives no tool
    executor, so destructive, write-capable, external, and approval-requiring
    tools are unavailable by construction. Candidate answers are used only by
    the evaluator and are never returned to the chat path.
    """

    def __init__(
        self,
        settings: AprilSettings,
        runtime_client: RuntimeEvalClient,
        *,
        judge_model_id: str | None,
        model_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.runtime_client = runtime_client
        self.judge_model_id = judge_model_id
        self.model_id = model_id

    async def evaluate(
        self,
        rollout: RolloutRecord,
        *,
        cancellation_event: asyncio.Event | None = None,
    ) -> ShadowMetrics:
        if rollout.candidate_type != "prompt_overlay":
            raise RolloutBlocked("lora_shadow_requires_separate_runtime_model_identity")
        if cancellation_event is not None and cancellation_event.is_set():
            raise asyncio.CancelledError
        candidate = Path(rollout.candidate_artifact_path).read_text(encoding="utf-8")
        baseline = (
            Path(rollout.baseline_artifact_path).read_text(encoding="utf-8")
            if rollout.baseline_artifact_path
            else ""
        )
        started = asyncio.get_running_loop().time()
        evaluation = await evaluate_overlay_candidate_real_runtime(
            agent=rollout.target_id,
            content=candidate,
            settings=self.settings,
            runtime_client=self.runtime_client,
            model_id=self.model_id,
            baseline_content=baseline,
            judge_model_id=self.judge_model_id,
        )
        elapsed_ms = max(0.0, (asyncio.get_running_loop().time() - started) * 1000.0)
        if cancellation_event is not None and cancellation_event.is_set():
            raise asyncio.CancelledError
        reviewed_cases = list_reviewed_eval_cases(self.settings)
        reviewed_ids = {str(item["case_id"]) for item in reviewed_cases}
        coding_ids = {
            str(item["case_id"])
            for item in reviewed_cases
            if str(item.get("case_type", "")).casefold()
            in {"coding", "coding_test", "code"}
        }
        tool_selection_ids = {
            str(item["case_id"])
            for item in reviewed_cases
            if item.get("expected_tool") is not None
            or str(item.get("case_type", "")).casefold() == "tool_selection"
        }
        reviewed_count = sum(
            1
            for outcome in evaluation.case_outcomes
            if str(outcome.get("case_id")) in reviewed_ids
        )
        baseline_passed = evaluation.baseline_cases_passed
        candidate_passed = evaluation.cases_passed
        samples = evaluation.cases_run
        coding_outcomes = [
            item
            for item in evaluation.case_outcomes
            if str(item.get("case_id")) in coding_ids
        ]
        tool_outcomes = [
            item
            for item in evaluation.case_outcomes
            if str(item.get("case_id")) in tool_selection_ids
        ]
        per_side_latency = elapsed_ms / max(1, samples * 2)
        return ShadowMetrics(
            sample_count=samples,
            human_reviewed_sample_count=reviewed_count,
            baseline_pass_count=baseline_passed,
            candidate_pass_count=candidate_passed,
            baseline_structured_valid_count=baseline_passed,
            candidate_structured_valid_count=candidate_passed,
            tool_selection_sample_count=len(tool_outcomes),
            baseline_tool_selection_correct_count=sum(
                int(bool(item.get("baseline_passed"))) for item in tool_outcomes
            ),
            candidate_tool_selection_correct_count=sum(
                int(bool(item.get("candidate_passed"))) for item in tool_outcomes
            ),
            coding_test_sample_count=len(coding_outcomes),
            baseline_coding_test_pass_count=sum(
                int(bool(item.get("baseline_passed"))) for item in coding_outcomes
            ),
            candidate_coding_test_pass_count=sum(
                int(bool(item.get("candidate_passed"))) for item in coding_outcomes
            ),
            baseline_failure_count=max(0, samples - baseline_passed),
            candidate_failure_count=max(0, samples - candidate_passed),
            baseline_latency_ms=per_side_latency,
            candidate_latency_ms=per_side_latency,
            baseline_compared=True,
            human_reviewed_evidence_present=reviewed_count > 0,
            hard_failure=evaluation.skipped,
        )


class RolloutService:
    def __init__(
        self,
        settings: AprilSettings,
        database: Database,
        *,
        audit: AuditLogger | None = None,
        guard: EvolutionWriteGuard | None = None,
        fault_hook: FaultHook | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.audit = audit
        self.guard = guard or EvolutionWriteGuard(settings, audit=audit)
        self.fault_hook = fault_hook

    async def create(
        self,
        *,
        candidate_type: CandidateType,
        target_id: str,
        candidate_id: str,
        candidate_artifact_path: Path,
        baseline_id: str | None = None,
        baseline_sha256: str | None = None,
        baseline_artifact_path: Path | None = None,
        minimum_samples: int | None = None,
        canary_fraction: float | None = None,
        canary_max_eligible_turns: int | None = None,
        rollout_id: str | None = None,
    ) -> RolloutRecord:
        self._require_rollouts_enabled()
        _validate_identifier(target_id, "target_id")
        _validate_identifier(candidate_id, "candidate_id")
        if candidate_type not in {"prompt_overlay", "lora_adapter"}:
            raise ValueError("unsupported_candidate_type")
        candidate_path = self._immutable_artifact(candidate_artifact_path)
        candidate_sha = _sha256_file(candidate_path)
        if candidate_type == "prompt_overlay":
            content = candidate_path.read_text(encoding="utf-8")
            reason = prompt_overlay_rejection_reason(
                content,
                max_chars=self.settings.evolution.prompt_overlay_max_chars,
            )
            if reason is not None:
                raise RolloutBlocked("candidate_overlay_policy_rejected")

        previous = await self._current_artifact(candidate_type, target_id)
        if baseline_id is None:
            baseline_id = str(previous["id"]) if previous is not None else f"stock:{target_id}"
        if baseline_artifact_path is None and previous is not None:
            raw_path = previous.get("path")
            baseline_artifact_path = Path(str(raw_path)) if raw_path else None
        normalized_baseline: Path | None = None
        if baseline_artifact_path is not None:
            normalized_baseline = self._immutable_artifact(baseline_artifact_path)
            actual_baseline_sha = _sha256_file(normalized_baseline)
            if baseline_sha256 is not None and baseline_sha256 != actual_baseline_sha:
                raise RolloutBlocked("baseline_hash_mismatch")
            baseline_sha256 = actual_baseline_sha
        if baseline_sha256 is None:
            baseline_sha256 = (
                str(previous["sha256"])
                if previous is not None
                else hashlib.sha256(b"").hexdigest()
            )
        _validate_sha256(baseline_sha256, "baseline_sha256")
        if candidate_sha == baseline_sha256:
            raise RolloutBlocked("candidate_matches_active_baseline")

        requested = minimum_samples or self.settings.evolution.rollout_shadow_min_samples
        if requested < 1:
            raise ValueError("minimum_samples_out_of_bounds")
        fraction = (
            canary_fraction
            if canary_fraction is not None
            else self.settings.evolution.rollout_canary_fraction
        )
        if not 0.0 < fraction <= 0.25:
            raise ValueError("canary_fraction_out_of_bounds")
        bounded_turns = (
            canary_max_eligible_turns
            if canary_max_eligible_turns is not None
            else self.settings.evolution.rollout_canary_max_eligible_turns
        )
        if bounded_turns < 1:
            raise ValueError("canary_turn_limit_out_of_bounds")
        configuration = {
            "schema_version": 1,
            "candidate_type": candidate_type,
            "target_id": target_id,
            "minimum_samples": requested,
            "canary_fraction": fraction,
            "canary_max_eligible_turns": bounded_turns,
            "canary_max_hours": self.settings.evolution.rollout_canary_max_hours,
            "max_pass_rate_regression": (
                self.settings.evolution.rollout_max_pass_rate_regression
            ),
            "max_structured_invalid_rate": (
                self.settings.evolution.rollout_max_structured_invalid_rate
            ),
            "max_failure_rate": self.settings.evolution.rollout_max_failure_rate,
            "max_latency_regression": (
                self.settings.evolution.rollout_max_latency_regression
            ),
            "max_fallback_rate": self.settings.evolution.rollout_max_fallback_rate,
            "automatic_promotion": False,
        }
        configuration_json = _canonical_json(configuration)
        configuration_sha = _sha256_text(configuration_json)
        identifier = rollout_id or str(uuid.uuid4())
        _validate_identifier(identifier, "rollout_id")
        now = utc_now_iso()
        previous_json = _canonical_json(previous) if previous is not None else None
        self.guard.validate_table("evolution_rollouts")
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO evolution_rollouts(
                    id, candidate_type, target_id, candidate_id,
                    candidate_sha256, candidate_artifact_path, baseline_id,
                    baseline_sha256, baseline_artifact_path, state,
                    configuration_json, configuration_sha256,
                    requested_minimum_samples, canary_traffic_fraction,
                    canary_max_eligible_turns, previous_active_artifact_json,
                    created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    candidate_type,
                    target_id,
                    candidate_id,
                    candidate_sha,
                    str(candidate_path),
                    baseline_id,
                    baseline_sha256,
                    str(normalized_baseline) if normalized_baseline is not None else None,
                    configuration_json,
                    configuration_sha,
                    requested,
                    fraction,
                    bounded_turns,
                    previous_json,
                    now,
                    now,
                ),
            )
            await self._event_tx(
                connection,
                identifier,
                "rollout_created",
                summary={
                    "candidate_type": candidate_type,
                    "candidate_sha256": candidate_sha,
                    "baseline_sha256": baseline_sha256,
                    "automatic_promotion": False,
                },
            )
        record = await self.require(identifier)
        self._audit("evolution_rollout_created", record)
        return record

    async def list(self, *, state: str | None = None) -> list[RolloutRecord]:
        if state is None:
            rows = await self.database.fetchall(
                "SELECT * FROM evolution_rollouts ORDER BY created_at DESC, id DESC"
            )
        else:
            if state not in _TRANSITIONS:
                raise ValueError("unknown_rollout_state")
            rows = await self.database.fetchall(
                """
                SELECT * FROM evolution_rollouts
                WHERE state = ?
                ORDER BY created_at DESC, id DESC
                """,
                (state,),
            )
        return [_record_from_row(row) for row in rows]

    async def require(self, rollout_id: str) -> RolloutRecord:
        row = await self.database.fetchone(
            "SELECT * FROM evolution_rollouts WHERE id = ?",
            (rollout_id,),
        )
        if row is None:
            raise RolloutBlocked("rollout_not_found")
        return _record_from_row(row)

    async def start_shadow(
        self,
        rollout_id: str,
        *,
        evaluator: ShadowEvaluator,
        cancellation_event: asyncio.Event | None = None,
    ) -> RolloutRecord:
        self._require_rollouts_enabled()
        record = await self.require(rollout_id)
        if record.state == "candidate":
            record = await self._transition(record, "shadow_pending")
        if record.state not in {"shadow_pending", "shadow_running"}:
            raise InvalidRolloutTransition(
                f"invalid_transition:{record.state}:shadow_running"
            )
        dataset_sha = record.shadow_dataset_sha256 or reviewed_dataset_hash(self.settings)
        if record.state == "shadow_pending":
            record = await self._transition(
                record,
                "shadow_running",
                updates={
                    "shadow_dataset_sha256": dataset_sha,
                    "started_at": record.started_at or utc_now_iso(),
                    "reason_code": None,
                },
            )
        self._audit("evolution_rollout_shadow_started", record)
        try:
            self._verify_candidate(record)
            metrics = await evaluator.evaluate(
                record,
                cancellation_event=cancellation_event,
            )
            if cancellation_event is not None and cancellation_event.is_set():
                raise asyncio.CancelledError
            if reviewed_dataset_hash(self.settings) != dataset_sha:
                raise RolloutBlocked("shadow_dataset_changed_during_evaluation")
            return await self.complete_shadow(rollout_id, metrics)
        except asyncio.CancelledError:
            return await self.cancel(rollout_id, reason_code="shadow_cancelled")
        except RolloutBlocked as exc:
            return await self.fail(rollout_id, reason_code=str(exc))
        except Exception:
            return await self.fail(rollout_id, reason_code="shadow_evaluator_failed")

    async def queue_shadow(self, rollout_id: str, *, store: Any) -> tuple[RolloutRecord, Any]:
        """Queue shadow A/B work in the existing durable background-job store."""

        self._require_rollouts_enabled()
        record = await self.require(rollout_id)
        if record.state == "candidate":
            record = await self._transition(record, "shadow_pending")
        if record.state != "shadow_pending":
            raise InvalidRolloutTransition(
                f"invalid_transition:{record.state}:shadow_pending"
            )
        job_id = f"rollout-shadow-{record.id}"
        if record.shadow_job_id is not None:
            return record, await store.require(record.shadow_job_id)
        job = await store.submit(
            job_type="evolution_shadow",
            payload={"rollout_id": record.id},
            owner="local-operator",
            job_id=job_id,
        )
        now = utc_now_iso()
        async with self.database.transaction() as connection:
            updated = await connection.execute(
                """
                UPDATE evolution_rollouts
                SET shadow_job_id = ?, updated_at = ?, version = version + 1
                WHERE id = ? AND state = 'shadow_pending' AND version = ?
                """,
                (job.id, now, record.id, record.version),
            )
            if updated.rowcount != 1:
                raise InvalidRolloutTransition("rollout_concurrency_conflict")
            await self._event_tx(
                connection,
                record.id,
                "shadow_job_queued",
                summary={"job_id": job.id},
            )
        return await self.require(record.id), job

    async def complete_shadow(
        self,
        rollout_id: str,
        metrics: ShadowMetrics,
    ) -> RolloutRecord:
        record = await self.require(rollout_id)
        if record.state != "shadow_running":
            raise InvalidRolloutTransition(
                f"invalid_transition:{record.state}:shadow_passed"
            )
        safe_metrics = metrics.safe_payload()
        gate_reason = self._shadow_gate(record, metrics)
        evidence = {
            "rollout_id": record.id,
            "candidate_sha256": record.candidate_sha256,
            "baseline_sha256": record.baseline_sha256,
            "configuration_sha256": record.configuration_sha256,
            "dataset_sha256": record.shadow_dataset_sha256,
            "metrics": safe_metrics,
        }
        evidence_sha = _sha256_text(_canonical_json(evidence))
        if gate_reason is not None:
            failed = await self._transition(
                record,
                "failed",
                updates={
                    "metrics_json": _canonical_json({"shadow": safe_metrics}),
                    "shadow_evidence_sha256": evidence_sha,
                    "completed_sample_count": metrics.sample_count,
                    "reason_code": gate_reason,
                    "completed_at": utc_now_iso(),
                },
            )
            self._audit("evolution_rollout_shadow_failed", failed, reason=gate_reason)
            return failed
        passed = await self._transition(
            record,
            "shadow_passed",
            updates={
                "metrics_json": _canonical_json({"shadow": safe_metrics}),
                "shadow_evidence_sha256": evidence_sha,
                "completed_sample_count": metrics.sample_count,
                "reason_code": None,
                "completed_at": utc_now_iso(),
            },
        )
        self._audit("evolution_rollout_shadow_passed", passed)
        return passed

    async def request_approval(
        self,
        rollout_id: str,
        *,
        stage: Literal["canary", "activation"],
        approvals: ApprovalStore,
        actor: str = "local-user",
        request_id: str | None = None,
    ) -> str:
        """Explicit owner action that creates, but never approves, an exact L4 gate."""

        record = await self.require(rollout_id)
        expected_state = "shadow_passed" if stage == "canary" else "canary_passed"
        if record.state != expected_state:
            raise RolloutBlocked(f"{stage}_approval_not_available_from_{record.state}")
        tool, args = self._approval_action(record, stage)
        response = await approvals.create(
            ApprovalRequest(
                tool=tool,
                args=args,
                agent="local-operator",
                permission_level=4,
                risk_level="system_action",
                affected_paths=[record.candidate_id],
                expected_side_effects=[
                    (
                        "route a bounded fraction of eligible low-risk prompt requests"
                        if stage == "canary"
                        else "publish the reviewed prompt overlay as active"
                    )
                ],
                metadata={
                    "rollout_id": record.id,
                    "stage": stage,
                    "candidate_sha256": record.candidate_sha256,
                },
            ),
            actor=actor,
            request_id=request_id or str(uuid.uuid4()),
        )
        return response.approval_id

    async def start_canary(
        self,
        rollout_id: str,
        *,
        approval_id: str,
    ) -> RolloutRecord:
        self._require_rollouts_enabled()
        if not self.settings.evolution.canary_enabled:
            raise RolloutBlocked("canary_disabled")
        record = await self.require(rollout_id)
        if record.candidate_type == "lora_adapter":
            raise RolloutBlocked("lora_canary_unsupported")
        if record.state == "shadow_passed":
            record = await self._transition(record, "canary_pending_approval")
        if record.state != "canary_pending_approval":
            raise InvalidRolloutTransition(
                f"invalid_transition:{record.state}:canary_running"
            )
        self._verify_candidate(record)
        self._verify_baseline(record)
        await self._verify_baseline_active(record)
        tool, args = self._approval_action(record, "canary")
        # Level 4 mutations fail closed if the hash-chained audit cannot accept
        # even the attempt record.
        self._audit("evolution_rollout_canary_start_requested", record)
        now = utc_now_iso()
        expires = (
            utc_now() + timedelta(hours=self.settings.evolution.rollout_canary_max_hours)
        ).isoformat().replace("+00:00", "Z")
        async with self.database.transaction() as connection:
            await self._validate_approval_tx(
                connection,
                approval_id=approval_id,
                tool=tool,
                args=args,
            )
            updated = await connection.execute(
                """
                UPDATE evolution_rollouts
                SET state = 'canary_running', canary_approval_id = ?,
                    canary_expires_at = ?, started_at = COALESCE(started_at, ?),
                    completed_sample_count = 0, completed_at = NULL,
                    updated_at = ?, reason_code = NULL,
                    version = version + 1
                WHERE id = ? AND state = 'canary_pending_approval' AND version = ?
                """,
                (approval_id, expires, now, now, record.id, record.version),
            )
            if updated.rowcount != 1:
                raise InvalidRolloutTransition("rollout_concurrency_conflict")
            consumed = await ApprovalStore.consume_in_transaction(
                connection,
                approval_id=approval_id,
                result={"ok": True, "rollout_id": record.id, "state": "canary_running"},
                consumed_at=now,
            )
            if not consumed:
                raise RolloutBlocked("approval_consumption_race")
            await self._event_tx(
                connection,
                record.id,
                "canary_started",
                summary={
                    "traffic_fraction": record.canary_traffic_fraction,
                    "max_eligible_turns": record.canary_max_eligible_turns,
                },
            )
        running = await self.require(record.id)
        try:
            self._audit("evolution_rollout_canary_started", running)
        except Exception:
            await self.rollback(
                running.id,
                reason_code="canary_audit_unavailable",
                automatic=True,
            )
            raise
        return running

    async def select_prompt_canary(
        self,
        *,
        target_id: str,
        context: CanaryContext,
    ) -> CanarySelection:
        if not (
            self.settings.evolution.enabled
            and self.settings.evolution.rollout_enabled
            and self.settings.evolution.canary_enabled
        ):
            return CanarySelection(None, False, False, "canary_disabled")
        row = await self.database.fetchone(
            """
            SELECT * FROM evolution_rollouts
            WHERE candidate_type = 'prompt_overlay'
              AND target_id = ?
              AND state = 'canary_running'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (target_id,),
        )
        if row is None:
            return CanarySelection(None, False, False, "no_active_canary")
        record = _record_from_row(row)
        expiry_reason = self._expiry_reason(record)
        if expiry_reason is not None:
            await self.rollback(record.id, reason_code=expiry_reason, automatic=True)
            return CanarySelection(record.id, False, False, expiry_reason)
        if not self._artifact_matches(
            Path(record.candidate_artifact_path), record.candidate_sha256
        ):
            await self.rollback(
                record.id,
                reason_code="candidate_artifact_unavailable_or_changed",
                automatic=True,
            )
            return CanarySelection(
                record.id,
                False,
                False,
                "candidate_artifact_unavailable_or_changed",
            )
        eligible, reason = _canary_eligible(context)
        request_hash = _sha256_text(context.stable_request_id)
        existing = await self.database.fetchone(
            """
            SELECT selected, eligible FROM evolution_rollout_assignments
            WHERE rollout_id = ? AND request_key_sha256 = ?
            """,
            (record.id, request_hash),
        )
        if existing is not None:
            selected = bool(existing["selected"])
            if not selected:
                return CanarySelection(
                    record.id,
                    False,
                    bool(existing["eligible"]),
                    "not_selected" if bool(existing["eligible"]) else reason,
                )
            return self._selected_overlay(record)

        if eligible and (
            record.canary_max_eligible_turns is not None
            and record.canary_eligible_turn_count >= record.canary_max_eligible_turns
        ):
            await self.rollback(
                record.id,
                reason_code="canary_turn_limit_insufficient_samples",
                automatic=True,
            )
            return CanarySelection(record.id, False, False, "canary_turn_limit_reached")
        bucket = int(
            hashlib.sha256(f"{record.id}:{context.stable_request_id}".encode()).hexdigest()[:16],
            16,
        ) / float(0xFFFFFFFFFFFFFFFF)
        selected = eligible and bucket < record.canary_traffic_fraction
        now = utc_now_iso()
        try:
            async with self.database.transaction() as connection:
                await connection.execute(
                    """
                    INSERT INTO evolution_rollout_assignments(
                        rollout_id, request_key_sha256, selected, eligible, created_at
                    )
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (record.id, request_hash, int(selected), int(eligible), now),
                )
                await connection.execute(
                    """
                    UPDATE evolution_rollouts
                    SET canary_eligible_turn_count =
                            canary_eligible_turn_count + ?,
                        canary_selected_turn_count =
                            canary_selected_turn_count + ?,
                        updated_at = ?, version = version + 1
                    WHERE id = ? AND state = 'canary_running'
                    """,
                    (int(eligible), int(selected), now, record.id),
                )
        except (sqlite3.IntegrityError, aiosqlite.IntegrityError):
            return await self.select_prompt_canary(target_id=target_id, context=context)
        if not selected:
            return CanarySelection(
                record.id,
                False,
                eligible,
                "not_selected" if eligible else reason,
            )
        return self._selected_overlay(record)

    async def record_canary_outcome_for_request(
        self,
        *,
        stable_request_id: str,
        outcome: dict[str, bool | int | float],
    ) -> RolloutRecord | None:
        request_hash = _sha256_text(stable_request_id)
        row = await self.database.fetchone(
            """
            SELECT rollout_id
            FROM evolution_rollout_assignments
            WHERE request_key_sha256 = ?
              AND selected = 1
              AND outcome_recorded = 0
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (request_hash,),
        )
        if row is None:
            return None
        return await self.record_canary_outcome(
            rollout_id=str(row["rollout_id"]),
            stable_request_id=stable_request_id,
            outcome=outcome,
        )

    async def rollout_for_request(self, stable_request_id: str) -> str | None:
        row = await self.database.fetchone(
            """
            SELECT rollout_id
            FROM evolution_rollout_assignments
            WHERE request_key_sha256 = ? AND selected = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (_sha256_text(stable_request_id),),
        )
        return str(row["rollout_id"]) if row is not None else None

    async def record_signal_for_agent_run(
        self,
        *,
        agent_run_id: str,
        signal: Literal[
            "approval_denied",
            "user_correction",
            "negative_feedback",
            "regeneration",
        ],
    ) -> RolloutRecord | None:
        row = await self.database.fetchone(
            "SELECT metadata_json FROM agent_runs WHERE id = ?",
            (agent_run_id,),
        )
        if row is None:
            return None
        try:
            metadata = json.loads(str(row["metadata_json"] or "{}"))
        except json.JSONDecodeError:
            return None
        if not isinstance(metadata, dict) or not isinstance(metadata.get("rollout_id"), str):
            return None
        rollout_id = str(metadata["rollout_id"])
        record = await self.require(rollout_id)
        if record.state not in {"canary_running", "active"}:
            return record
        aggregate = dict(record.metrics)
        canary = dict(aggregate.get("canary") or {})
        field_name = {
            "approval_denied": "approval_denial_count",
            "user_correction": "user_correction_count",
            "negative_feedback": "negative_feedback_count",
            "regeneration": "regeneration_count",
        }[signal]
        canary[field_name] = int(canary.get(field_name, 0)) + 1
        canary["failure_count"] = int(canary.get("failure_count", 0)) + 1
        aggregate["canary"] = canary
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                UPDATE evolution_rollouts
                SET metrics_json = ?, updated_at = ?, version = version + 1
                WHERE id = ? AND state IN ('canary_running', 'active')
                """,
                (_canonical_json(aggregate), utc_now_iso(), rollout_id),
            )
            await self._event_tx(
                connection,
                rollout_id,
                "post_outcome_signal",
                summary={"signal": signal},
            )
        updated = await self.require(rollout_id)
        reason = self._canary_regression_reason(updated)
        if reason is not None:
            return await self.rollback(updated.id, reason_code=reason, automatic=True)
        return updated

    async def track_active_request(
        self,
        *,
        target_id: str,
        context: CanaryContext,
    ) -> str | None:
        """Bind a newly-active rollout to safe post-activation monitoring."""

        row = await self.database.fetchone(
            """
            SELECT * FROM evolution_rollouts
            WHERE candidate_type = 'prompt_overlay'
              AND target_id = ?
              AND state = 'active'
            ORDER BY completed_at DESC, created_at DESC
            LIMIT 1
            """,
            (target_id,),
        )
        if row is None:
            return None
        record = _record_from_row(row)
        if not self._artifact_matches(
            Path(record.candidate_artifact_path), record.candidate_sha256
        ):
            await self.rollback(
                record.id,
                reason_code="candidate_artifact_unavailable_or_changed",
                automatic=True,
            )
            return None
        request_hash = _sha256_text(context.stable_request_id)
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT OR IGNORE INTO evolution_rollout_assignments(
                    rollout_id, request_key_sha256, selected, eligible, created_at
                )
                VALUES(?, ?, 1, 1, ?)
                """,
                (record.id, request_hash, utc_now_iso()),
            )
        return record.id

    async def record_canary_outcome(
        self,
        *,
        rollout_id: str,
        stable_request_id: str,
        outcome: dict[str, bool | int | float],
    ) -> RolloutRecord:
        safe = _validate_safe_outcome(outcome)
        request_hash = _sha256_text(stable_request_id)
        now = utc_now_iso()
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                SELECT selected, outcome_recorded
                FROM evolution_rollout_assignments
                WHERE rollout_id = ? AND request_key_sha256 = ?
                """,
                (rollout_id, request_hash),
            )
            assignment = await cursor.fetchone()
            if assignment is None or not bool(assignment["selected"]):
                raise RolloutBlocked("canary_assignment_not_found")
            if bool(assignment["outcome_recorded"]):
                return await self.require(rollout_id)
            cursor = await connection.execute(
                "SELECT * FROM evolution_rollouts WHERE id = ?",
                (rollout_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise RolloutBlocked("rollout_not_found")
            record = _record_from_row(row)
            if record.state not in {"canary_running", "active"}:
                raise RolloutBlocked("rollout_not_monitoring_outcomes")
            aggregate = dict(record.metrics)
            canary = dict(aggregate.get("canary") or {})
            _aggregate_outcome(canary, safe)
            aggregate["canary"] = canary
            await connection.execute(
                """
                UPDATE evolution_rollout_assignments
                SET outcome_recorded = 1, safe_outcome_json = ?,
                    completed_at = ?
                WHERE rollout_id = ? AND request_key_sha256 = ?
                  AND outcome_recorded = 0
                """,
                (_canonical_json(safe), now, rollout_id, request_hash),
            )
            await connection.execute(
                """
                UPDATE evolution_rollouts
                SET metrics_json = ?, completed_sample_count =
                        completed_sample_count + 1,
                    updated_at = ?, version = version + 1
                WHERE id = ?
                """,
                (_canonical_json(aggregate), now, rollout_id),
            )
            await self._event_tx(
                connection,
                rollout_id,
                "canary_outcome_recorded",
                summary=_outcome_event_summary(safe),
            )
        updated = await self.require(rollout_id)
        reason = self._canary_regression_reason(updated)
        if reason is not None:
            return await self.rollback(updated.id, reason_code=reason, automatic=True)
        if (
            updated.state == "canary_running"
            and updated.completed_sample_count
            >= self.settings.evolution.rollout_canary_min_samples
        ):
            passed = await self._transition(
                updated,
                "canary_passed",
                updates={"completed_at": utc_now_iso(), "reason_code": None},
            )
            self._audit("evolution_rollout_canary_passed", passed)
            return passed
        return updated

    async def promote(
        self,
        rollout_id: str,
        *,
        approval_id: str,
        readiness: PromotionReadiness,
        cancellation_event: asyncio.Event | None = None,
    ) -> RolloutRecord:
        self._require_rollouts_enabled()
        record = await self.require(rollout_id)
        if record.candidate_type == "lora_adapter":
            raise RolloutBlocked("lora_canary_unsupported")
        if record.state != "canary_passed":
            raise InvalidRolloutTransition(
                f"invalid_transition:{record.state}:activation_pending_approval"
            )
        self._promotion_gate(record, readiness)
        await self._verify_baseline_active(record)
        tool, args = self._approval_action(record, "activation")
        self._audit("evolution_rollout_activation_requested", record)
        async with self.database.transaction() as connection:
            await self._validate_approval_tx(
                connection,
                approval_id=approval_id,
                tool=tool,
                args=args,
            )
        record = await self._transition(
            record,
            "activation_pending_approval",
            updates={
                "activation_approval_id": approval_id,
                "transition_phase": "activation_prepared",
                "completed_at": None,
            },
        )
        await self._fault("activation_prepared", record)
        if cancellation_event is not None and cancellation_event.is_set():
            return await self.rollback(
                record.id,
                reason_code="activation_cancelled",
                automatic=True,
            )
        self._verify_candidate(record)
        self._verify_baseline(record)
        now = utc_now_iso()
        # Prompt publication and rollout finalization are intentionally separate
        # durable phases. Startup reconciliation treats the gap as unsafe and
        # restores the exact previous artifact.
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE prompt_versions SET active = 0 WHERE agent = ?",
                (record.target_id,),
            )
            cursor = await connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next_version
                FROM prompt_versions WHERE agent = ?
                """,
                (record.target_id,),
            )
            next_row = await cursor.fetchone()
            version = int(next_row["next_version"]) if next_row is not None else 1
            artifact_id = f"{record.target_id}:{version}"
            await connection.execute(
                """
                INSERT INTO prompt_versions(
                    id, agent, version, overlay_path, content_hash, active,
                    eval_score, baseline_score, created_at
                )
                VALUES(?, ?, ?, ?, ?, 1, NULL, NULL, ?)
                """,
                (
                    artifact_id,
                    record.target_id,
                    version,
                    record.candidate_artifact_path,
                    record.candidate_sha256,
                    now,
                ),
            )
            updated = await connection.execute(
                """
                UPDATE evolution_rollouts
                SET transition_phase = 'pointer_published', updated_at = ?,
                    version = version + 1
                WHERE id = ? AND state = 'activation_pending_approval'
                  AND version = ?
                """,
                (now, record.id, record.version),
            )
            if updated.rowcount != 1:
                raise InvalidRolloutTransition("rollout_concurrency_conflict")
            await self._event_tx(
                connection,
                record.id,
                "activation_pointer_published",
                summary={"artifact_id": artifact_id, "sha256": record.candidate_sha256},
            )
        published = await self.require(record.id)
        await self._fault("pointer_published", published)
        if cancellation_event is not None and cancellation_event.is_set():
            return await self.rollback(
                published.id,
                reason_code="activation_cancelled",
                automatic=True,
            )
        async with self.database.transaction() as connection:
            await self._validate_approval_tx(
                connection,
                approval_id=approval_id,
                tool=tool,
                args=args,
            )
            consumed = await ApprovalStore.consume_in_transaction(
                connection,
                approval_id=approval_id,
                result={"ok": True, "rollout_id": record.id, "state": "active"},
                consumed_at=now,
            )
            if not consumed:
                raise RolloutBlocked("approval_consumption_race")
            updated = await connection.execute(
                """
                UPDATE evolution_rollouts
                SET state = 'active', transition_phase = NULL,
                    activation_approval_id = ?, completed_at = ?,
                    updated_at = ?, reason_code = NULL, version = version + 1
                WHERE id = ? AND state = 'activation_pending_approval'
                  AND transition_phase = 'pointer_published'
                  AND version = ?
                """,
                (approval_id, now, now, record.id, published.version),
            )
            if updated.rowcount != 1:
                raise InvalidRolloutTransition("rollout_concurrency_conflict")
            await self._event_tx(connection, record.id, "rollout_activated")
        active = await self.require(record.id)
        try:
            self._audit("evolution_rollout_activated", active)
        except Exception:
            await self.rollback(
                active.id,
                reason_code="activation_audit_unavailable",
                automatic=True,
            )
            raise
        return active

    async def cancel(
        self,
        rollout_id: str,
        *,
        reason_code: str = "operator_cancelled",
    ) -> RolloutRecord:
        record = await self.require(rollout_id)
        if record.state in TERMINAL_STATES:
            return record
        if record.state == "active" or record.transition_phase == "pointer_published":
            return await self.rollback(
                rollout_id,
                reason_code=reason_code,
                automatic=False,
            )
        cancelled = await self._transition(
            record,
            "cancelled",
            updates={"reason_code": _reason_code(reason_code), "completed_at": utc_now_iso()},
        )
        self._audit("evolution_rollout_cancelled", cancelled, reason=reason_code)
        return cancelled

    async def fail(self, rollout_id: str, *, reason_code: str) -> RolloutRecord:
        record = await self.require(rollout_id)
        if record.state in TERMINAL_STATES:
            return record
        if record.state == "active" or record.transition_phase == "pointer_published":
            return await self.rollback(
                rollout_id,
                reason_code=reason_code,
                automatic=True,
            )
        failed = await self._transition(
            record,
            "failed",
            updates={"reason_code": _reason_code(reason_code), "completed_at": utc_now_iso()},
        )
        self._audit("evolution_rollout_failed", failed, reason=reason_code)
        return failed

    async def rollback(
        self,
        rollout_id: str,
        *,
        reason_code: str = "operator_rollback",
        automatic: bool = False,
    ) -> RolloutRecord:
        record = await self.require(rollout_id)
        if record.state == "rolled_back":
            return record
        if record.state in {"failed", "cancelled", "rejected"}:
            return record
        if record.candidate_type == "lora_adapter":
            # A LoRA rollout cannot reach canary/active with the current Runtime,
            # so there is no global pointer to mutate here.
            rolled = await self._transition(
                record,
                "rolled_back",
                updates={
                    "reason_code": _reason_code(reason_code),
                    "rolled_back_at": utc_now_iso(),
                    "completed_at": utc_now_iso(),
                    "transition_phase": None,
                },
            )
            self._audit(
                "evolution_rollout_rolled_back",
                rolled,
                reason=reason_code,
                automatic=automatic,
            )
            return rolled

        now = utc_now_iso()
        previous = record.previous_active_artifact
        if previous is not None and not self._previous_artifact_available(previous):
            async with self.database.transaction() as connection:
                await connection.execute(
                    """
                    UPDATE evolution_rollouts
                    SET state = 'failed', reason_code = 'rollback_previous_unavailable',
                        transition_phase = 'rollback_required', updated_at = ?,
                        completed_at = ?, version = version + 1
                    WHERE id = ?
                    """,
                    (now, now, record.id),
                )
                await self._event_tx(
                    connection,
                    record.id,
                    "rollback_failed",
                    reason_code="rollback_previous_unavailable",
                )
            failed = await self.require(record.id)
            self._audit(
                "evolution_rollout_rollback_failed",
                failed,
                reason="rollback_previous_unavailable",
                automatic=automatic,
            )
            return failed
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE prompt_versions SET active = 0 WHERE agent = ?",
                (record.target_id,),
            )
            if previous is not None:
                restored = await connection.execute(
                    """
                    UPDATE prompt_versions
                    SET active = 1
                    WHERE id = ? AND agent = ? AND content_hash = ?
                    """,
                    (
                        str(previous["id"]),
                        record.target_id,
                        str(previous["sha256"]),
                    ),
                )
                if restored.rowcount != 1:
                    raise RolloutBlocked("rollback_previous_database_record_missing")
            updated = await connection.execute(
                """
                UPDATE evolution_rollouts
                SET state = 'rolled_back', reason_code = ?,
                    transition_phase = NULL, rolled_back_at = ?,
                    completed_at = ?, updated_at = ?, version = version + 1
                WHERE id = ? AND state NOT IN (
                    'rolled_back', 'failed', 'cancelled', 'rejected'
                )
                """,
                (_reason_code(reason_code), now, now, now, record.id),
            )
            if updated.rowcount != 1:
                latest = await self.require(record.id)
                return latest
            await self._event_tx(
                connection,
                record.id,
                "rollout_rolled_back",
                reason_code=_reason_code(reason_code),
                summary={"automatic": automatic},
            )
        rolled = await self.require(record.id)
        self._audit(
            "evolution_rollout_rolled_back",
            rolled,
            reason=reason_code,
            automatic=automatic,
        )
        return rolled

    async def reconcile_startup(self) -> dict[str, Any]:
        """Fail closed on interrupted publication and unsafe pointer drift."""

        rows = await self.database.fetchall(
            """
            SELECT * FROM evolution_rollouts
            WHERE transition_phase IS NOT NULL
               OR state IN (
                   'canary_running', 'activation_pending_approval', 'active'
               )
            ORDER BY created_at, id
            """
        )
        reconciled = 0
        rollback_required: list[str] = []
        for row in rows:
            record = _record_from_row(row)
            reason: str | None = None
            if record.transition_phase in {"activation_prepared", "pointer_published"}:
                reason = "startup_incomplete_activation"
            elif record.state == "canary_running":
                reason = self._expiry_reason(record)
                if reason is None and not self._artifact_matches(
                    Path(record.candidate_artifact_path),
                    record.candidate_sha256,
                ):
                    reason = "candidate_artifact_unavailable_or_changed"
                if reason is None and await self._candidate_is_active(record):
                    reason = "canary_pointer_database_disagreement"
            elif record.state == "active":
                if not self._artifact_matches(
                    Path(record.candidate_artifact_path),
                    record.candidate_sha256,
                ):
                    reason = "candidate_artifact_unavailable_or_changed"
                elif not await self._candidate_is_active(record):
                    reason = "active_pointer_database_disagreement"
            if reason is None:
                continue
            result = await self.rollback(record.id, reason_code=reason, automatic=True)
            reconciled += 1
            if result.state != "rolled_back":
                rollback_required.append(record.id)
        return {
            "reconciled_rollout_count": reconciled,
            "rollback_required_rollout_ids": rollback_required,
            "healthy": not rollback_required,
        }

    async def health(self) -> dict[str, Any]:
        records = await self.list()
        active_canaries = [item for item in records if item.state == "canary_running"]
        expired = [
            item for item in active_canaries if self._expiry_reason(item) is not None
        ]
        incomplete = [item for item in records if item.transition_phase is not None]
        mismatches: list[str] = []
        unavailable: list[str] = []
        hash_mismatches: list[str] = []
        rollback_required: list[str] = []
        for item in records:
            if item.state in TERMINAL_STATES and item.transition_phase != "rollback_required":
                continue
            candidate_path = Path(item.candidate_artifact_path)
            if not candidate_path.is_file() or candidate_path.is_symlink():
                unavailable.append(item.id)
            elif not self._artifact_matches(candidate_path, item.candidate_sha256):
                hash_mismatches.append(item.id)
            if item.state == "active" and not await self._candidate_is_active(item):
                mismatches.append(item.id)
            if item.transition_phase == "rollback_required":
                rollback_required.append(item.id)
        unsafe = bool(
            incomplete
            or expired
            or unavailable
            or hash_mismatches
            or mismatches
            or rollback_required
        )
        return {
            "enabled": bool(
                self.settings.evolution.enabled
                and self.settings.evolution.rollout_enabled
            ),
            "canary_enabled": self.settings.evolution.canary_enabled,
            "automatic_candidate_creation": (
                self.settings.evolution.automatic_candidate_creation
            ),
            "automatic_promotion": self.settings.evolution.automatic_promotion,
            "status": (
                "degraded"
                if unsafe
                else ("disabled" if not self.settings.evolution.rollout_enabled else "ok")
            ),
            "incomplete_transition_count": len(incomplete),
            "active_canary_count": len(active_canaries),
            "expired_canary_count": len(expired),
            "candidate_unavailable_count": len(unavailable),
            "candidate_hash_mismatch_count": len(hash_mismatches),
            "pointer_database_disagreement_count": len(mismatches),
            "rollback_required_count": len(rollback_required),
            "lora_canary_supported": False,
            "lora_canary_readiness_reason": "lora_canary_unsupported",
            "action": (
                "run april evolve rollout status ROLLOUT_ID, then rollback"
                if unsafe
                else None
            ),
        }

    async def _transition(
        self,
        record: RolloutRecord,
        target: RolloutState,
        *,
        updates: dict[str, Any] | None = None,
    ) -> RolloutRecord:
        if target not in _TRANSITIONS[record.state]:
            raise InvalidRolloutTransition(
                f"invalid_transition:{record.state}:{target}"
            )
        values = dict(updates or {})
        values["state"] = target
        values["updated_at"] = utc_now_iso()
        assignments = [f"{column} = ?" for column in values]
        parameters = [
            _encode_column_value(column, value) for column, value in values.items()
        ]
        parameters.extend([record.id, record.state, record.version])
        async with self.database.transaction() as connection:
            updated = await connection.execute(
                f"""
                UPDATE evolution_rollouts
                SET {", ".join(assignments)}, version = version + 1
                WHERE id = ? AND state = ? AND version = ?
                """,
                tuple(parameters),
            )
            if updated.rowcount != 1:
                raise InvalidRolloutTransition("rollout_concurrency_conflict")
            await self._event_tx(
                connection,
                record.id,
                "state_transition",
                summary={"from": record.state, "to": target},
            )
        return await self.require(record.id)

    def _shadow_gate(
        self,
        record: RolloutRecord,
        metrics: ShadowMetrics,
    ) -> str | None:
        if metrics.training_metric_only:
            return "training_metrics_are_not_shadow_evidence"
        if not metrics.baseline_compared:
            return "shadow_baseline_comparison_missing"
        if not metrics.human_reviewed_evidence_present:
            return "human_reviewed_evidence_missing"
        if metrics.human_reviewed_sample_count < 1:
            return "human_reviewed_evidence_missing"
        if metrics.sample_count < record.requested_minimum_samples:
            return "shadow_minimum_samples_not_met"
        if metrics.hard_failure:
            return "shadow_hard_failure"
        if metrics.candidate_pass_count < metrics.baseline_pass_count:
            return "shadow_pass_rate_regression"
        if (
            metrics.candidate_structured_valid_count
            < metrics.baseline_structured_valid_count
        ):
            return "shadow_structured_output_regression"
        if (
            metrics.tool_selection_sample_count > 0
            and metrics.candidate_tool_selection_correct_count
            < metrics.baseline_tool_selection_correct_count
        ):
            return "shadow_tool_selection_regression"
        if (
            metrics.coding_test_sample_count > 0
            and metrics.candidate_coding_test_pass_count
            < metrics.baseline_coding_test_pass_count
        ):
            return "shadow_coding_test_regression"
        if metrics.candidate_failure_count > metrics.baseline_failure_count:
            return "shadow_failure_rate_regression"
        if metrics.baseline_latency_ms > 0 and (
            metrics.candidate_latency_ms
            > metrics.baseline_latency_ms
            * (1.0 + self.settings.evolution.rollout_max_latency_regression)
        ):
            return "shadow_latency_regression"
        return None

    def _canary_regression_reason(self, record: RolloutRecord) -> str | None:
        canary = record.metrics.get("canary")
        if not isinstance(canary, dict):
            return None
        samples = int(canary.get("sample_count", 0))
        if samples < 1:
            return None
        if int(canary.get("hard_failure_count", 0)) > 0:
            return "canary_hard_failure"
        failures = int(canary.get("failure_count", 0))
        if failures / samples > self.settings.evolution.rollout_max_failure_rate:
            return "canary_failure_rate_threshold"
        invalid = int(canary.get("structured_invalid_count", 0))
        if invalid / samples > (
            self.settings.evolution.rollout_max_structured_invalid_rate
        ):
            return "canary_structured_invalid_threshold"
        fallbacks = int(canary.get("fallback_count", 0))
        if fallbacks / samples > self.settings.evolution.rollout_max_fallback_rate:
            return "canary_fallback_threshold"
        baseline_latency = float(canary.get("baseline_latency_ms_total", 0.0))
        candidate_latency = float(canary.get("latency_ms_total", 0.0))
        if baseline_latency > 0 and candidate_latency > baseline_latency * (
            1.0 + self.settings.evolution.rollout_max_latency_regression
        ):
            return "canary_latency_regression"
        shadow = record.metrics.get("shadow")
        if isinstance(shadow, dict):
            shadow_samples = max(1, int(shadow.get("sample_count", 0)))
            baseline_rate = int(shadow.get("baseline_pass_count", 0)) / shadow_samples
            successes = int(canary.get("success_count", 0))
            candidate_rate = successes / samples
            if candidate_rate + self.settings.evolution.rollout_max_pass_rate_regression < (
                baseline_rate
            ):
                return "canary_pass_rate_regression"
        return None

    def _promotion_gate(
        self,
        record: RolloutRecord,
        readiness: PromotionReadiness,
    ) -> None:
        shadow = record.metrics.get("shadow")
        if not isinstance(shadow, dict):
            raise RolloutBlocked("shadow_evidence_missing")
        if not bool(shadow.get("human_reviewed_evidence_present")):
            raise RolloutBlocked("human_reviewed_evidence_missing")
        if record.completed_sample_count < (
            self.settings.evolution.rollout_canary_min_samples
        ):
            raise RolloutBlocked("canary_minimum_samples_not_met")
        reason = self._canary_regression_reason(record)
        if reason is not None:
            raise RolloutBlocked(reason)
        if not readiness.runtime_healthy:
            raise RolloutBlocked("runtime_readiness_unhealthy")
        if not readiness.database_healthy:
            raise RolloutBlocked("database_readiness_unhealthy")
        self._verify_candidate(record)
        self._verify_baseline(record)

    async def _validate_approval_tx(
        self,
        connection: Any,
        *,
        approval_id: str,
        tool: str,
        args: dict[str, Any],
    ) -> None:
        cursor = await connection.execute(
            "SELECT * FROM approvals WHERE id = ?",
            (approval_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RolloutBlocked("approval_not_found")
        try:
            stored_args = json.loads(str(row["args_json"]))
            metadata = json.loads(str(row["metadata_json"] or "{}"))
        except json.JSONDecodeError as exc:
            raise RolloutBlocked("approval_record_invalid") from exc
        if not isinstance(stored_args, dict) or not isinstance(metadata, dict):
            raise RolloutBlocked("approval_record_invalid")
        if (
            str(row["tool"]) != tool
            or stored_args != args
            or str(row["canonical_hash"]) != canonical_hash(tool, args, metadata)
            or int(row["permission_level"]) != 4
            or str(row["risk_level"]) != "system_action"
        ):
            raise RolloutBlocked("approval_action_mismatch")
        if str(row["status"]) != "approved":
            raise RolloutBlocked("approval_not_approved")
        try:
            if parse_utc_iso(str(row["expires_at"])) < utc_now():
                raise RolloutBlocked("approval_expired")
        except ValueError as exc:
            raise RolloutBlocked("approval_record_invalid") from exc

    def _approval_action(
        self,
        record: RolloutRecord,
        stage: Literal["canary", "activation"],
    ) -> tuple[str, dict[str, Any]]:
        return (
            f"evolution_rollout_{stage}",
            {
                "rollout_id": record.id,
                "candidate_id": record.candidate_id,
                "candidate_sha256": record.candidate_sha256,
                "baseline_id": record.baseline_id,
                "baseline_sha256": record.baseline_sha256,
                "configuration_sha256": record.configuration_sha256,
                "shadow_evidence_sha256": record.shadow_evidence_sha256,
            },
        )

    def _selected_overlay(self, record: RolloutRecord) -> CanarySelection:
        try:
            self._verify_candidate(record)
            text = Path(record.candidate_artifact_path).read_text(encoding="utf-8")
        except (OSError, RolloutBlocked):
            return CanarySelection(
                record.id,
                False,
                True,
                "candidate_artifact_unavailable_or_changed",
            )
        return CanarySelection(record.id, True, True, "selected", text)

    def _verify_candidate(self, record: RolloutRecord) -> None:
        if not self._artifact_matches(
            Path(record.candidate_artifact_path),
            record.candidate_sha256,
        ):
            raise RolloutBlocked("candidate_artifact_unavailable_or_changed")

    def _verify_baseline(self, record: RolloutRecord) -> None:
        if record.baseline_artifact_path is not None and not self._artifact_matches(
            Path(record.baseline_artifact_path),
            record.baseline_sha256,
        ):
            raise RolloutBlocked("baseline_artifact_unavailable_or_changed")

    async def _verify_baseline_active(self, record: RolloutRecord) -> None:
        current = await self._current_artifact(record.candidate_type, record.target_id)
        previous = record.previous_active_artifact
        if previous is None:
            if current is not None:
                raise RolloutBlocked("baseline_active_pointer_changed")
            return
        if current is None or (
            str(current.get("id")) != str(previous.get("id"))
            or str(current.get("sha256")) != str(previous.get("sha256"))
        ):
            raise RolloutBlocked("baseline_active_pointer_changed")

    def _immutable_artifact(self, path: Path) -> Path:
        normalized = self.guard.validate_path(path)
        if not normalized.is_file() or normalized.is_symlink():
            raise RolloutBlocked("artifact_missing_or_not_regular_file")
        return normalized

    @staticmethod
    def _artifact_matches(path: Path, expected_sha256: str) -> bool:
        try:
            return (
                path.is_file()
                and not path.is_symlink()
                and _sha256_file(path) == expected_sha256
            )
        except OSError:
            return False

    async def _current_artifact(
        self,
        candidate_type: CandidateType,
        target_id: str,
    ) -> dict[str, Any] | None:
        if candidate_type == "lora_adapter":
            row = await self.database.fetchone(
                """
                SELECT id, adapter_path, created_at
                FROM model_adapters
                WHERE model_id = ? AND status = 'active'
                """,
                (target_id,),
            )
            if row is None:
                return None
            path = Path(str(row["adapter_path"])).resolve(strict=False)
            if not path.is_file():
                raise RolloutBlocked("active_baseline_artifact_unavailable")
            return {
                "id": str(row["id"]),
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        row = await self.database.fetchone(
            """
            SELECT id, version, overlay_path, content_hash
            FROM prompt_versions
            WHERE agent = ? AND active = 1
            """,
            (target_id,),
        )
        if row is None:
            return None
        path = Path(str(row["overlay_path"])).resolve(strict=False)
        if not self._artifact_matches(path, str(row["content_hash"])):
            raise RolloutBlocked("active_baseline_artifact_unavailable_or_changed")
        return {
            "id": str(row["id"]),
            "version": int(row["version"]),
            "path": str(path),
            "sha256": str(row["content_hash"]),
        }

    async def _candidate_is_active(self, record: RolloutRecord) -> bool:
        if record.candidate_type != "prompt_overlay":
            return False
        row = await self.database.fetchone(
            """
            SELECT content_hash FROM prompt_versions
            WHERE agent = ? AND active = 1
            """,
            (record.target_id,),
        )
        return row is not None and str(row["content_hash"]) == record.candidate_sha256

    def _previous_artifact_available(self, previous: dict[str, Any]) -> bool:
        path = previous.get("path")
        sha = str(previous.get("sha256") or "")
        return bool(path) and _SHA256_RE.fullmatch(sha) is not None and self._artifact_matches(
            Path(str(path)), sha
        )

    def _expiry_reason(self, record: RolloutRecord) -> str | None:
        if record.canary_expires_at is None:
            return None
        try:
            expired = parse_utc_iso(record.canary_expires_at) <= utc_now()
        except ValueError:
            return "canary_expiry_invalid"
        if not expired:
            return None
        if record.completed_sample_count < (
            self.settings.evolution.rollout_canary_min_samples
        ):
            return "canary_expired_insufficient_samples"
        return "canary_expired"

    async def _fault(self, phase: str, record: RolloutRecord) -> None:
        if self.fault_hook is None:
            return
        result = self.fault_hook(phase, record)
        if result is not None:
            await result

    async def _event_tx(
        self,
        connection: Any,
        rollout_id: str,
        event_type: str,
        *,
        reason_code: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        self.guard.validate_table("evolution_rollout_events")
        await connection.execute(
            """
            INSERT INTO evolution_rollout_events(
                rollout_id, event_type, reason_code, safe_summary_json, created_at
            )
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                rollout_id,
                event_type[:96],
                _reason_code(reason_code) if reason_code else None,
                _canonical_json(summary or {}),
                utc_now_iso(),
            ),
        )

    def _require_rollouts_enabled(self) -> None:
        if not self.settings.evolution.enabled:
            raise RolloutBlocked("evolution_disabled")
        if not self.settings.evolution.rollout_enabled:
            raise RolloutBlocked("rollout_disabled")
        if self.settings.evolution.automatic_candidate_creation:
            # The production implementation intentionally supports explicit
            # creation only. A future automatic creator must be separately
            # reviewed rather than silently piggybacking on this service.
            raise RolloutBlocked("automatic_candidate_creation_not_supported")
        if self.settings.evolution.automatic_promotion:
            raise RolloutBlocked("automatic_promotion_not_supported")

    def _audit(
        self,
        event_type: str,
        record: RolloutRecord,
        *,
        reason: str | None = None,
        automatic: bool = False,
    ) -> None:
        if self.audit is None:
            return
        self.audit.write(
            {
                "event_type": event_type,
                "actor": "april-core" if automatic else "local-user",
                "rollout_id": record.id,
                "candidate_type": record.candidate_type,
                "candidate_id": record.candidate_id,
                "candidate_sha256": record.candidate_sha256,
                "baseline_id": record.baseline_id,
                "baseline_sha256": record.baseline_sha256,
                "state": record.state,
                "reason_code": _reason_code(reason) if reason else None,
                "automatic": automatic,
            }
        )


def reviewed_dataset_hash(settings: AprilSettings) -> str:
    """Hash reviewed immutable case bytes without persisting their content."""

    directory = settings.evolution_path / "evals" / "reviewed"
    parts: list[str] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.yaml")):
            if path.is_file() and not path.is_symlink():
                parts.append(f"{path.name}:{_sha256_file(path)}")
    return _sha256_text("\n".join(parts))


def inspect_rollout_state(settings: AprilSettings) -> dict[str, Any]:
    """Read-only, redaction-safe rollout probe for offline readiness."""

    enabled = bool(settings.evolution.enabled and settings.evolution.rollout_enabled)
    base = {
        "enabled": enabled,
        "canary_enabled": settings.evolution.canary_enabled,
        "automatic_candidate_creation": settings.evolution.automatic_candidate_creation,
        "automatic_promotion": settings.evolution.automatic_promotion,
        "lora_canary_supported": False,
        "lora_canary_readiness_reason": "lora_canary_unsupported",
        "incomplete_transition_count": 0,
        "active_canary_count": 0,
        "expired_canary_count": 0,
        "candidate_unavailable_count": 0,
        "candidate_hash_mismatch_count": 0,
        "pointer_database_disagreement_count": 0,
        "rollback_required_count": 0,
    }
    database_path = settings.database_path.expanduser().resolve(strict=False)
    if not database_path.is_file():
        return {**base, "status": "disabled" if not enabled else "not_initialized"}
    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'evolution_rollouts'
            """
        ).fetchone()
        if table is None:
            return {**base, "status": "disabled" if not enabled else "not_initialized"}
        rows = list(connection.execute("SELECT * FROM evolution_rollouts"))
        active_prompt = {
            str(row["agent"]): str(row["content_hash"])
            for row in connection.execute(
                "SELECT agent, content_hash FROM prompt_versions WHERE active = 1"
            )
        }
    finally:
        connection.close()
    incomplete = 0
    active_canary = 0
    expired = 0
    unavailable = 0
    mismatch = 0
    disagreement = 0
    rollback_required = 0
    now = utc_now()
    for row in rows:
        state = str(row["state"])
        phase = str(row["transition_phase"]) if row["transition_phase"] is not None else None
        if state in TERMINAL_STATES and phase != "rollback_required":
            continue
        if phase is not None:
            incomplete += 1
        if phase == "rollback_required":
            rollback_required += 1
        if state == "canary_running":
            active_canary += 1
            expiry = row["canary_expires_at"]
            if expiry is not None:
                try:
                    expired += int(parse_utc_iso(str(expiry)) <= now)
                except ValueError:
                    expired += 1
        path = Path(str(row["candidate_artifact_path"]))
        if not path.is_file() or path.is_symlink():
            unavailable += 1
        elif _sha256_file(path) != str(row["candidate_sha256"]):
            mismatch += 1
        if (
            state == "active"
            and str(row["candidate_type"]) == "prompt_overlay"
            and active_prompt.get(str(row["target_id"]))
            != str(row["candidate_sha256"])
        ):
            disagreement += 1
        if (
            state == "canary_running"
            and str(row["candidate_type"]) == "prompt_overlay"
            and active_prompt.get(str(row["target_id"]))
            == str(row["candidate_sha256"])
        ):
            disagreement += 1
    unsafe = bool(
        incomplete
        or expired
        or unavailable
        or mismatch
        or disagreement
        or rollback_required
    )
    status = "degraded" if unsafe else ("disabled" if not enabled else "ok")
    return {
        **base,
        "status": status,
        "incomplete_transition_count": incomplete,
        "active_canary_count": active_canary,
        "expired_canary_count": expired,
        "candidate_unavailable_count": unavailable,
        "candidate_hash_mismatch_count": mismatch,
        "pointer_database_disagreement_count": disagreement,
        "rollback_required_count": rollback_required,
        "action": (
            "run april evolve rollout list, then inspect and rollback unsafe rollouts"
            if unsafe
            else None
        ),
    }


def _record_from_row(row: Any) -> RolloutRecord:
    try:
        metrics = json.loads(str(row["metrics_json"] or "{}"))
        previous = (
            json.loads(str(row["previous_active_artifact_json"]))
            if row["previous_active_artifact_json"] is not None
            else None
        )
    except json.JSONDecodeError as exc:
        raise RolloutBlocked("rollout_record_invalid") from exc
    if not isinstance(metrics, dict) or (previous is not None and not isinstance(previous, dict)):
        raise RolloutBlocked("rollout_record_invalid")
    return RolloutRecord(
        id=str(row["id"]),
        candidate_type=str(row["candidate_type"]),  # type: ignore[arg-type]
        target_id=str(row["target_id"]),
        candidate_id=str(row["candidate_id"]),
        candidate_sha256=str(row["candidate_sha256"]),
        candidate_artifact_path=str(row["candidate_artifact_path"]),
        baseline_id=str(row["baseline_id"]),
        baseline_sha256=str(row["baseline_sha256"]),
        baseline_artifact_path=(
            str(row["baseline_artifact_path"])
            if row["baseline_artifact_path"] is not None
            else None
        ),
        state=str(row["state"]),  # type: ignore[arg-type]
        configuration_sha256=str(row["configuration_sha256"]),
        shadow_dataset_sha256=(
            str(row["shadow_dataset_sha256"])
            if row["shadow_dataset_sha256"] is not None
            else None
        ),
        shadow_evidence_sha256=(
            str(row["shadow_evidence_sha256"])
            if row["shadow_evidence_sha256"] is not None
            else None
        ),
        requested_minimum_samples=int(row["requested_minimum_samples"]),
        completed_sample_count=int(row["completed_sample_count"]),
        canary_traffic_fraction=float(row["canary_traffic_fraction"]),
        canary_max_eligible_turns=(
            int(row["canary_max_eligible_turns"])
            if row["canary_max_eligible_turns"] is not None
            else None
        ),
        canary_eligible_turn_count=int(row["canary_eligible_turn_count"]),
        canary_selected_turn_count=int(row["canary_selected_turn_count"]),
        canary_expires_at=(
            str(row["canary_expires_at"]) if row["canary_expires_at"] is not None else None
        ),
        metrics=metrics,
        reason_code=str(row["reason_code"]) if row["reason_code"] is not None else None,
        canary_approval_id=(
            str(row["canary_approval_id"]) if row["canary_approval_id"] is not None else None
        ),
        activation_approval_id=(
            str(row["activation_approval_id"])
            if row["activation_approval_id"] is not None
            else None
        ),
        previous_active_artifact=previous,
        transition_phase=(
            str(row["transition_phase"]) if row["transition_phase"] is not None else None
        ),
        shadow_job_id=(
            str(row["shadow_job_id"]) if row["shadow_job_id"] is not None else None
        ),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=str(row["started_at"]) if row["started_at"] is not None else None,
        completed_at=(
            str(row["completed_at"]) if row["completed_at"] is not None else None
        ),
        rolled_back_at=(
            str(row["rolled_back_at"]) if row["rolled_back_at"] is not None else None
        ),
        version=int(row["version"]),
    )


def _canary_eligible(context: CanaryContext) -> tuple[bool, str]:
    if context.source in {"voice", "wake", "background", "dreamer"} or context.live_voice:
        return False, "live_or_background_source_excluded"
    if context.mode != "standard" or context.high_risk_reasoning:
        return False, "high_risk_reasoning_excluded"
    if context.permission_level >= 3:
        return False, "approval_requiring_interaction_excluded"
    if context.risk_level not in {"none", "read_only"}:
        return False, "write_or_external_risk_excluded"
    if context.agent in {"coding_agent", "system_action_agent"}:
        return False, "write_capable_agent_excluded"
    if context.has_pending_approval:
        return False, "pending_approval_excluded"
    if (
        context.destructive
        or context.external_side_effect
        or context.security_sensitive
        or context.database_write
        or context.repository_write
        or context.background_evolution
    ):
        return False, "unsafe_interaction_excluded"
    read_only_tools = {
        "read_file",
        "search_files",
        "list_files",
        "git_status",
        "git_diff",
        "git_log",
        "git_branch",
        "repo_indexer",
    }
    if any(tool not in read_only_tools for tool in context.tool_names):
        return False, "non_read_only_tool_excluded"
    return True, "eligible"


def _validate_safe_outcome(
    outcome: dict[str, bool | int | float],
) -> dict[str, bool | int | float]:
    if set(outcome) - _SAFE_OUTCOME_KEYS:
        raise ValueError("unsafe_rollout_outcome_key")
    safe: dict[str, bool | int | float] = {}
    for key, value in outcome.items():
        if key in {"latency_ms", "baseline_latency_ms"}:
            parsed = float(value)
            if parsed < 0 or parsed > 3_600_000:
                raise ValueError("rollout_latency_out_of_bounds")
            safe[key] = parsed
        elif isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, int) and value in {0, 1}:
            safe[key] = bool(value)
        else:
            raise ValueError("rollout_outcome_must_be_boolean_or_bounded_latency")
    return safe


def _aggregate_outcome(
    aggregate: dict[str, Any],
    outcome: dict[str, bool | int | float],
) -> None:
    aggregate["sample_count"] = int(aggregate.get("sample_count", 0)) + 1
    aggregate["success_count"] = int(aggregate.get("success_count", 0)) + int(
        bool(outcome.get("success", True))
    )
    failure = bool(
        outcome.get("tool_failure")
        or outcome.get("coding_test_failed")
        or outcome.get("runtime_failure")
        or outcome.get("approval_denied")
        or outcome.get("user_correction")
        or outcome.get("negative_feedback")
        or outcome.get("regeneration")
        or not bool(outcome.get("success", True))
    )
    aggregate["failure_count"] = int(aggregate.get("failure_count", 0)) + int(failure)
    aggregate["structured_invalid_count"] = int(
        aggregate.get("structured_invalid_count", 0)
    ) + int(outcome.get("structured_output_valid") is False)
    aggregate["repair_count"] = int(aggregate.get("repair_count", 0)) + int(
        bool(outcome.get("repair_attempted"))
    )
    aggregate["tool_success_count"] = int(aggregate.get("tool_success_count", 0)) + int(
        bool(outcome.get("tool_success"))
    )
    aggregate["tool_failure_count"] = int(aggregate.get("tool_failure_count", 0)) + int(
        bool(outcome.get("tool_failure"))
    )
    for key, field_name in (
        ("approval_denied", "approval_denial_count"),
        ("user_correction", "user_correction_count"),
        ("negative_feedback", "negative_feedback_count"),
        ("regeneration", "regeneration_count"),
        ("coding_test_passed", "coding_test_pass_count"),
        ("coding_test_failed", "coding_test_failure_count"),
        ("runtime_failure", "runtime_failure_count"),
        ("candidate_fallback", "fallback_count"),
        ("hard_failure", "hard_failure_count"),
    ):
        aggregate[field_name] = int(aggregate.get(field_name, 0)) + int(
            bool(outcome.get(key))
        )
    aggregate["latency_ms_total"] = float(aggregate.get("latency_ms_total", 0.0)) + float(
        outcome.get("latency_ms", 0.0)
    )
    aggregate["baseline_latency_ms_total"] = float(
        aggregate.get("baseline_latency_ms_total", 0.0)
    ) + float(outcome.get("baseline_latency_ms", 0.0))


def _outcome_event_summary(
    outcome: dict[str, bool | int | float],
) -> dict[str, Any]:
    return {
        "hard_failure": bool(outcome.get("hard_failure")),
        "runtime_failure": bool(outcome.get("runtime_failure")),
        "candidate_fallback": bool(outcome.get("candidate_fallback")),
        "success": bool(outcome.get("success", True)),
    }


def _encode_column_value(column: str, value: Any) -> Any:
    if column in {"metrics_json", "previous_active_artifact_json"} and not isinstance(
        value, str
    ):
        return _canonical_json(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, field: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field}_invalid")


def _validate_identifier(value: str, field: str) -> None:
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field}_invalid")


def _reason_code(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = "_".join(str(value).casefold().split())
    safe = "".join(char for char in normalized if char.isalnum() or char == "_")
    return safe[:160] or "unknown"
