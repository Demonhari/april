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


class RolloutPersistence(RolloutServiceBase):
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
                str(previous["sha256"]) if previous is not None else hashlib.sha256(b"").hexdigest()
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
            "max_pass_rate_regression": (self.settings.evolution.rollout_max_pass_rate_regression),
            "max_structured_invalid_rate": (
                self.settings.evolution.rollout_max_structured_invalid_rate
            ),
            "max_failure_rate": self.settings.evolution.rollout_max_failure_rate,
            "max_latency_regression": (self.settings.evolution.rollout_max_latency_regression),
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
