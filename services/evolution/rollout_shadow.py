from __future__ import annotations

# ruff: noqa: F401
# mypy: disable-error-code="attr-defined"
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
from services.evolution.evaluator import RuntimeEvalClient, evaluate_overlay_candidate_real_runtime
from services.evolution.rollout_base import RolloutServiceBase
from services.evolution.rollout_evaluation import reviewed_dataset_hash
from services.evolution.rollout_models import (
    _IDENTIFIER_RE,
    _SAFE_OUTCOME_KEYS,
    _SHA256_RE,
    _TRANSITIONS,
    TERMINAL_STATES,
    CanaryContext,
    CanarySelection,
    CandidateType,
    FaultHook,
    InvalidRolloutTransition,
    PromotionReadiness,
    RolloutBlocked,
    RolloutError,
    RolloutRecord,
    RolloutState,
    ShadowEvaluator,
    ShadowMetrics,
)
from services.evolution.rollout_policy import (
    _aggregate_outcome,
    _canary_eligible,
    _canonical_json,
    _encode_column_value,
    _outcome_event_summary,
    _reason_code,
    _sha256_file,
    _sha256_text,
    _validate_identifier,
    _validate_safe_outcome,
    _validate_sha256,
)
from services.evolution.rollout_records import _record_from_row
from services.evolution.versions import prompt_overlay_rejection_reason
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database
from services.permissions.approvals import ApprovalStore, canonical_hash
from services.permissions.schemas import ApprovalRequest


class RolloutShadow(RolloutServiceBase):
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
            raise InvalidRolloutTransition(f"invalid_transition:{record.state}:shadow_running")
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
            raise InvalidRolloutTransition(f"invalid_transition:{record.state}:shadow_pending")
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
            raise InvalidRolloutTransition(f"invalid_transition:{record.state}:shadow_passed")
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
