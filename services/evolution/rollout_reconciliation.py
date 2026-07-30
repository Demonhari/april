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


class RolloutReconciliation(RolloutServiceBase):
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
        expired = [item for item in active_canaries if self._expiry_reason(item) is not None]
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
                self.settings.evolution.enabled and self.settings.evolution.rollout_enabled
            ),
            "canary_enabled": self.settings.evolution.canary_enabled,
            "automatic_candidate_creation": (self.settings.evolution.automatic_candidate_creation),
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
                "run april evolve rollout status ROLLOUT_ID, then rollback" if unsafe else None
            ),
        }
