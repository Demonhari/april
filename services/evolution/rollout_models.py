from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

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
    "shadow_passed": frozenset({"canary_pending_approval", "cancelled", "rejected", "failed"}),
    "canary_pending_approval": frozenset({"canary_running", "cancelled", "rejected", "failed"}),
    "canary_running": frozenset({"canary_passed", "cancelled", "rolled_back", "failed"}),
    "canary_passed": frozenset(
        {"activation_pending_approval", "cancelled", "rolled_back", "failed"}
    ),
    "activation_pending_approval": frozenset({"active", "cancelled", "rolled_back", "failed"}),
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
                key: value for key, value in previous.items() if key in {"id", "version", "sha256"}
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
