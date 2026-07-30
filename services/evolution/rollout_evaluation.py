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
            if str(item.get("case_type", "")).casefold() in {"coding", "coding_test", "code"}
        }
        tool_selection_ids = {
            str(item["case_id"])
            for item in reviewed_cases
            if item.get("expected_tool") is not None
            or str(item.get("case_type", "")).casefold() == "tool_selection"
        }
        reviewed_count = sum(
            1 for outcome in evaluation.case_outcomes if str(outcome.get("case_id")) in reviewed_ids
        )
        baseline_passed = evaluation.baseline_cases_passed
        candidate_passed = evaluation.cases_passed
        samples = evaluation.cases_run
        coding_outcomes = [
            item for item in evaluation.case_outcomes if str(item.get("case_id")) in coding_ids
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


def reviewed_dataset_hash(settings: AprilSettings) -> str:
    """Hash reviewed immutable case bytes without persisting their content."""

    directory = settings.evolution_path / "evals" / "reviewed"
    parts: list[str] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.yaml")):
            if path.is_file() and not path.is_symlink():
                parts.append(f"{path.name}:{_sha256_file(path)}")
    return _sha256_text("\n".join(parts))
