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


class RolloutActivation(RolloutServiceBase):
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
            return await self._promote_lora(
                record,
                approval_id=approval_id,
                readiness=readiness,
            )
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

    async def _promote_lora(
        self,
        record: RolloutRecord,
        *,
        approval_id: str,
        readiness: PromotionReadiness,
    ) -> RolloutRecord:
        if record.state != "canary_passed":
            raise InvalidRolloutTransition(
                f"invalid_transition:{record.state}:activation_pending_approval"
            )
        self._promotion_gate(record, readiness)
        runtime = await self.database.fetchone(
            "SELECT * FROM evolution_rollout_runtime WHERE rollout_id = ?",
            (record.id,),
        )
        if runtime is None:
            raise RolloutBlocked("candidate_runtime_identity_missing")
        if (
            str(runtime["status"]) != "loaded"
            or str(runtime["integrity_state"]) != "verified"
            or str(runtime["adapter_sha256"]) != record.candidate_sha256
            or str(runtime["configuration_sha256"]) != record.configuration_sha256
        ):
            raise RolloutBlocked("candidate_runtime_integrity_mismatch")
        tool, args = self._approval_action(record, "activation")
        self._audit("evolution_rollout_activation_requested", record)
        now = utc_now_iso()
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
                WHERE id = ? AND state = 'canary_passed' AND version = ?
                """,
                (approval_id, now, now, record.id, record.version),
            )
            if updated.rowcount != 1:
                raise InvalidRolloutTransition("rollout_concurrency_conflict")
            await self._event_tx(
                connection,
                record.id,
                "rollout_activated",
                summary={"runtime_instance_id": str(runtime["instance_id"])},
            )
        active = await self.require(record.id)
        self._audit("evolution_rollout_activated", active)
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
        if (
            record.state in {"failed", "cancelled", "rejected"}
            and record.transition_phase != "rollback_required"
        ):
            return record
        if record.candidate_type == "lora_adapter":
            runtime = await self.database.fetchone(
                "SELECT * FROM evolution_rollout_runtime WHERE rollout_id = ?",
                (record.id,),
            )
            if runtime is not None and str(runtime["status"]) in {"loaded", "rollback_required"}:
                client = self.runtime_client
                if client is None:
                    now = utc_now_iso()
                    async with self.database.transaction() as connection:
                        await connection.execute(
                            """
                            UPDATE evolution_rollouts
                            SET state = 'failed', transition_phase = 'rollback_required',
                                reason_code = ?, updated_at = ?, version = version + 1
                            WHERE id = ?
                            """,
                            ("rollback_runtime_unavailable", now, record.id),
                        )
                        await self._event_tx(
                            connection,
                            record.id,
                            "candidate_runtime_rollback_required",
                            reason_code="rollback_runtime_unavailable",
                        )
                    return await self.require(record.id)
                try:
                    await client.unload_candidate(instance_id=str(runtime["instance_id"]))
                except Exception:
                    now = utc_now_iso()
                    async with self.database.transaction() as connection:
                        await connection.execute(
                            """
                            UPDATE evolution_rollouts
                            SET state = 'failed', transition_phase = 'rollback_required',
                                reason_code = ?, updated_at = ?, version = version + 1
                            WHERE id = ?
                            """,
                            ("rollback_runtime_unavailable", now, record.id),
                        )
                        await self._event_tx(
                            connection,
                            record.id,
                            "candidate_runtime_rollback_required",
                            reason_code="rollback_runtime_unavailable",
                        )
                    return await self.require(record.id)
                now = utc_now_iso()
                async with self.database.transaction() as connection:
                    await connection.execute(
                        """
                        UPDATE evolution_rollout_runtime
                        SET status = 'unloaded', updated_at = ?
                        WHERE rollout_id = ?
                        """,
                        (now, record.id),
                    )
                    await self._event_tx(
                        connection,
                        record.id,
                        "candidate_runtime_unloaded",
                        summary={"instance_id": str(runtime["instance_id"])},
                    )
                self._audit(
                    "evolution_candidate_runtime_unloaded",
                    record,
                    reason=reason_code,
                    automatic=automatic,
                )
            if record.state == "failed":
                now = utc_now_iso()
                async with self.database.transaction() as connection:
                    updated = await connection.execute(
                        """
                        UPDATE evolution_rollouts
                        SET state = 'rolled_back', reason_code = ?,
                            rolled_back_at = ?, completed_at = ?,
                            transition_phase = NULL, updated_at = ?, version = version + 1
                        WHERE id = ? AND state = 'failed' AND transition_phase = 'rollback_required'
                        """,
                        (_reason_code(reason_code), now, now, now, record.id),
                    )
                    if updated.rowcount != 1:
                        return await self.require(record.id)
                    await self._event_tx(
                        connection,
                        record.id,
                        "rollout_rolled_back",
                        reason_code=_reason_code(reason_code),
                        summary={"automatic": automatic},
                    )
                rolled = await self.require(record.id)
            else:
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
