from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agents.schemas import AgentName

RiskLevel = Literal[
    "none", "read_only", "safe_write", "code_write", "system_action", "external_action"
]

# Canonical fixture and policy vocabulary. Legacy spellings are accepted only
# at the compatibility boundary below and are normalized before downstream use.
CanonicalIntent = Literal[
    "normal_conversation",
    "planning",
    "coding_repo_analysis",
    "document_reading",
    "creative_writing",
    "deep_reasoning",
    "memory_lookup",
    "memory_write",
    "patch_proposal",
    "code_modification",
    "command_execution",
    "log_cleanup",
    "package_install",
    "external_action",
    "ambiguous_request",
    "prompt_injection",
    "path_escape_attempt",
    "sensitive_content",
    "unsupported_tool",
    "reminders",
]

_LEGACY_INTENT_ALIASES = {
    "repository_search": "coding_repo_analysis",
    "configured_test_execution": "command_execution",
    "reminder_list": "reminders",
    "reminder_cancel": "reminders",
    "reminder_create": "reminders",
    "destructive_action": "sensitive_content",
    "destructive": "sensitive_content",
    "path_escape": "path_escape_attempt",
    "unknown_tool": "unsupported_tool",
    "approval_command": "normal_conversation",
    "rejection_command": "normal_conversation",
    "direct_agent_run": "normal_conversation",
    # Benchmark fixture category aliases are normalized only after the strict
    # JSON boundary; the generated schema still exposes the canonical 20.
    "git_status": "coding_repo_analysis",
    "git_diff": "coding_repo_analysis",
    "file_reading": "document_reading",
    "file_search": "coding_repo_analysis",
    "reminder_creation": "reminders",
    "reminder_listing": "reminders",
    "patch_preparation": "patch_proposal",
    "test_execution": "command_execution",
    "approval": "normal_conversation",
    "rejection": "normal_conversation",
    "destructive_external": "external_action",
    "ambiguous_general": "ambiguous_request",
}


class PlannedToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class BrainDecision(BaseModel):
    intent: CanonicalIntent
    agent: AgentName
    model_id: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    high_stakes: bool = False
    tools_needed: list[str] = Field(default_factory=list)
    planned_tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    memory_queries: list[str] = Field(default_factory=list)
    permission_level: int = Field(ge=0, le=5)
    risk_level: RiskLevel
    needs_confirmation: bool
    task_steps: list[str] = Field(default_factory=list, max_length=8)
    decision_summary: str
    routing_method: Literal["model", "model_repair", "fallback"] = "model"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_intent(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("intent"), str):
            value = dict(value)
            value["intent"] = _LEGACY_INTENT_ALIASES.get(value["intent"], value["intent"])
        return value


class RouteSource(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    MODEL_REPAIR = "model_repair"
    FALLBACK = "fallback"


class RouteResult(BaseModel):
    """Trusted routing provenance kept outside model-generated output."""

    decision: BrainDecision
    route_source: RouteSource
    raw_model_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    historical_reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    effective_confidence: float = Field(ge=0.0, le=1.0)
    reliability_sample_count: int = Field(default=0, ge=0)
    confidence_source: str
    matched_rule: str | None = None
    fallback_reason: str | None = None
    structured_output_valid: bool = True
    repair_used: bool = False

    @property
    def route_key(self) -> str:
        tool_class = (
            self.decision.planned_tool_calls[0].tool
            if self.decision.planned_tool_calls
            else (self.decision.tools_needed[0] if self.decision.tools_needed else "no_tool")
        )
        return ":".join(
            (
                self.decision.intent[:64],
                self.decision.agent,
                tool_class[:64],
            )
        )
