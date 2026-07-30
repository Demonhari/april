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


class RolloutSupport(RolloutServiceBase):
    async def _transition(
        self,
        record: RolloutRecord,
        target: RolloutState,
        *,
        updates: dict[str, Any] | None = None,
    ) -> RolloutRecord:
        if target not in _TRANSITIONS[record.state]:
            raise InvalidRolloutTransition(f"invalid_transition:{record.state}:{target}")
        values = dict(updates or {})
        values["state"] = target
        values["updated_at"] = utc_now_iso()
        assignments = [f"{column} = ?" for column in values]
        parameters = [_encode_column_value(column, value) for column, value in values.items()]
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
        if metrics.candidate_structured_valid_count < metrics.baseline_structured_valid_count:
            return "shadow_structured_output_regression"
        if (
            metrics.tool_selection_sample_count > 0
            and metrics.candidate_tool_selection_correct_count
            < metrics.baseline_tool_selection_correct_count
        ):
            return "shadow_tool_selection_regression"
        if (
            metrics.coding_test_sample_count > 0
            and metrics.candidate_coding_test_pass_count < metrics.baseline_coding_test_pass_count
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
        if invalid / samples > (self.settings.evolution.rollout_max_structured_invalid_rate):
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
        if record.completed_sample_count < (self.settings.evolution.rollout_canary_min_samples):
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
                path.is_file() and not path.is_symlink() and _sha256_file(path) == expected_sha256
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
        return (
            bool(path)
            and _SHA256_RE.fullmatch(sha) is not None
            and self._artifact_matches(Path(str(path)), sha)
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
        if record.completed_sample_count < (self.settings.evolution.rollout_canary_min_samples):
            return "canary_expired_insufficient_samples"
        return "canary_expired"

    async def _fault(self, phase: str, record: RolloutRecord) -> None:
        if self.fault_hook is None:
            return
        result = self.fault_hook(phase, record)
        if result is not None:
            await result

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
