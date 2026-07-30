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


class RolloutCanary(RolloutServiceBase):
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
            raise InvalidRolloutTransition(f"invalid_transition:{record.state}:canary_running")
        self._verify_candidate(record)
        self._verify_baseline(record)
        await self._verify_baseline_active(record)
        tool, args = self._approval_action(record, "canary")
        # Level 4 mutations fail closed if the hash-chained audit cannot accept
        # even the attempt record.
        self._audit("evolution_rollout_canary_start_requested", record)
        now = utc_now_iso()
        expires = (
            (utc_now() + timedelta(hours=self.settings.evolution.rollout_canary_max_hours))
            .isoformat()
            .replace("+00:00", "Z")
        )
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
            and updated.completed_sample_count >= self.settings.evolution.rollout_canary_min_samples
        ):
            passed = await self._transition(
                updated,
                "canary_passed",
                updates={"completed_at": utc_now_iso(), "reason_code": None},
            )
            self._audit("evolution_rollout_canary_passed", passed)
            return passed
        return updated
