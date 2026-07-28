from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.schemas import AgentName

RiskLevel = Literal[
    "none", "read_only", "safe_write", "code_write", "system_action", "external_action"
]


class PlannedToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class BrainDecision(BaseModel):
    intent: str
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
