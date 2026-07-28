from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from april_common.settings import BrainSettings
from april_common.time import parse_utc_iso, utc_now_iso
from services.brain.schemas import RouteResult, RouteSource
from services.memory.database import Database


@dataclass(frozen=True, slots=True)
class ReliabilityEstimate:
    historical_reliability: float
    effective_confidence: float
    sample_count: int
    confidence_source: str


class RoutingReliabilityService:
    """Outcome-based, bounded calibration with a neutral Beta prior."""

    def __init__(self, database: Database, settings: BrainSettings) -> None:
        self.database = database
        self.minimum_samples = settings.routing_reliability_min_samples
        self.prior_successes = settings.routing_reliability_prior_successes
        self.prior_failures = settings.routing_reliability_prior_failures

    async def calibrate(self, route: RouteResult) -> RouteResult:
        if route.route_source is RouteSource.DETERMINISTIC:
            return route.model_copy(
                update={
                    "historical_reliability": None,
                    "effective_confidence": 1.0,
                    "reliability_sample_count": 0,
                    "confidence_source": "deterministic_rule",
                }
            )
        estimate = await self.estimate(
            route_key=route.route_key,
            raw_confidence=route.raw_model_confidence,
        )
        return route.model_copy(
            update={
                "historical_reliability": estimate.historical_reliability,
                "effective_confidence": estimate.effective_confidence,
                "reliability_sample_count": estimate.sample_count,
                "confidence_source": estimate.confidence_source,
            }
        )

    async def estimate(
        self,
        *,
        route_key: str,
        raw_confidence: float | None,
    ) -> ReliabilityEstimate:
        rows = await self.database.fetchall(
            """
            SELECT structured_output_valid, repair_used, tool_outcome,
                   approval_outcome, user_correction, negative_feedback,
                   regeneration_or_retry, coding_test_outcome, final_status,
                   created_at
            FROM routing_outcomes
            WHERE route_key = ?
            ORDER BY created_at DESC
            LIMIT 200
            """,
            (route_key,),
        )
        successes = self.prior_successes
        failures = self.prior_failures
        now = datetime.now(UTC)
        for row in rows:
            weight = self._recency_weight(str(row["created_at"]), now)
            score = self._outcome_score(dict(row))
            successes += weight * score
            failures += weight * (1.0 - score)
        reliability = self._bounded(successes / (successes + failures))
        raw = 0.5 if raw_confidence is None else self._bounded(raw_confidence)
        if len(rows) < self.minimum_samples:
            # A neutral prior is reported, but cold-start uncertainty must not
            # manufacture Deep/verified escalation for otherwise normal turns.
            effective = raw
            source = "neutral_prior_insufficient_history"
        else:
            effective = 0.4 * raw + 0.6 * reliability
            source = "model_plus_outcome_history"
        return ReliabilityEstimate(
            historical_reliability=reliability,
            effective_confidence=self._bounded(effective),
            sample_count=len(rows),
            confidence_source=source,
        )

    async def record(
        self,
        route: RouteResult,
        *,
        agent_run_id: str | None,
        final_status: str,
        tool_outcome: str | None = None,
        approval_outcome: str | None = None,
        user_correction: bool = False,
        negative_feedback: bool = False,
        regeneration_or_retry: bool = False,
        coding_test_outcome: str | None = None,
    ) -> str:
        outcome_id = str(uuid.uuid4())
        now = utc_now_iso()
        decision = route.decision
        await self.database.execute(
            """
            INSERT INTO routing_outcomes(
                id, agent_run_id, route_key, intent, agent, route_source,
                normalized_tool_class, raw_confidence, effective_confidence,
                structured_output_valid, repair_used, tool_outcome,
                approval_outcome, user_correction, negative_feedback,
                regeneration_or_retry, coding_test_outcome, final_status,
                created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome_id,
                agent_run_id,
                route.route_key,
                decision.intent[:64],
                decision.agent,
                route.route_source.value,
                self._tool_class(route),
                route.raw_model_confidence,
                route.effective_confidence,
                int(route.structured_output_valid),
                int(route.repair_used),
                self._bounded_category(tool_outcome),
                self._bounded_category(approval_outcome),
                int(user_correction),
                int(negative_feedback),
                int(regeneration_or_retry),
                self._bounded_category(coding_test_outcome),
                self._bounded_category(final_status) or "unknown",
                now,
                now,
            ),
        )
        return outcome_id

    async def mark_approval_outcome(
        self,
        *,
        agent_run_id: str,
        outcome: str,
        final_status: str | None = None,
    ) -> None:
        now = utc_now_iso()
        if final_status is None:
            await self.database.execute(
                """
                UPDATE routing_outcomes
                SET approval_outcome = ?, updated_at = ?
                WHERE agent_run_id = ?
                """,
                (self._bounded_category(outcome), now, agent_run_id),
            )
            return
        await self.database.execute(
            """
            UPDATE routing_outcomes
            SET approval_outcome = ?, final_status = ?, updated_at = ?
            WHERE agent_run_id = ?
            """,
            (
                self._bounded_category(outcome),
                self._bounded_category(final_status),
                now,
                agent_run_id,
            ),
        )

    async def mark_negative_feedback(
        self,
        *,
        agent_run_id: str,
        implicit_correction: bool = False,
    ) -> None:
        if implicit_correction:
            statement = """
                UPDATE routing_outcomes
                SET user_correction = 1, updated_at = ?
                WHERE agent_run_id = ?
            """
        else:
            statement = """
                UPDATE routing_outcomes
                SET negative_feedback = 1, updated_at = ?
                WHERE agent_run_id = ?
            """
        await self.database.execute(statement, (utc_now_iso(), agent_run_id))

    async def mark_latest_route_outcome(
        self,
        *,
        route_key: str,
        approval_outcome: str,
        tool_outcome: str | None = None,
        coding_test_outcome: str | None = None,
        final_status: str | None = None,
    ) -> None:
        await self.database.execute(
            """
            UPDATE routing_outcomes
            SET approval_outcome = ?,
                tool_outcome = COALESCE(?, tool_outcome),
                coding_test_outcome = COALESCE(?, coding_test_outcome),
                final_status = COALESCE(?, final_status),
                updated_at = ?
            WHERE id = (
                SELECT id FROM routing_outcomes
                WHERE route_key = ?
                ORDER BY created_at DESC
                LIMIT 1
            )
            """,
            (
                self._bounded_category(approval_outcome),
                self._bounded_category(tool_outcome),
                self._bounded_category(coding_test_outcome),
                self._bounded_category(final_status),
                utc_now_iso(),
                route_key[:194],
            ),
        )

    @staticmethod
    def _outcome_score(row: dict[str, Any]) -> float:
        status = str(row.get("final_status") or "").lower()
        score = 1.0 if status in {"ok", "completed", "success", "pending_approval"} else 0.0
        if not bool(row.get("structured_output_valid")):
            score = min(score, 0.1)
        if bool(row.get("repair_used")):
            score = min(score, 0.75)
        if str(row.get("tool_outcome") or "").lower() in {"failed", "error"}:
            score = 0.0
        if str(row.get("approval_outcome") or "").lower() in {"denied", "expired"}:
            score = min(score, 0.25)
        if bool(row.get("user_correction")) or bool(row.get("negative_feedback")):
            score = 0.0
        if bool(row.get("regeneration_or_retry")):
            score = min(score, 0.25)
        test = str(row.get("coding_test_outcome") or "").lower()
        if test in {"failed", "error"}:
            score = 0.0
        elif test in {"passed", "success"}:
            score = max(score, 1.0)
        return RoutingReliabilityService._bounded(score)

    @staticmethod
    def _recency_weight(created_at: str, now: datetime) -> float:
        try:
            age_days = max(0.0, (now - parse_utc_iso(created_at)).total_seconds() / 86400)
        except (TypeError, ValueError):
            return 0.5
        return max(0.125, math.pow(0.5, age_days / 90.0))

    @staticmethod
    def _tool_class(route: RouteResult) -> str:
        tools = [
            *[call.tool for call in route.decision.planned_tool_calls],
            *route.decision.tools_needed,
        ]
        tool = tools[0] if tools else "no_tool"
        if tool.startswith("git_"):
            return "git_read"
        if tool in {"read_file", "search_files", "list_files", "repo_indexer"}:
            return "local_read"
        if tool in {"patch_generator", "patch_applier", "write_file"}:
            return "code_change"
        if tool in {"run_command", "test_runner"}:
            return "test_or_command"
        if tool in {"create_reminder", "list_reminders", "cancel_reminder"}:
            return "reminder"
        return tool[:64]

    @staticmethod
    def _bounded(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    @staticmethod
    def _bounded_category(value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().lower()[:64]
