from __future__ import annotations

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
    tools_needed: list[str] = Field(default_factory=list)
    planned_tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    memory_queries: list[str] = Field(default_factory=list)
    permission_level: int = Field(ge=0, le=5)
    risk_level: RiskLevel
    needs_confirmation: bool
    task_steps: list[str] = Field(default_factory=list, max_length=8)
    decision_summary: str
    routing_method: Literal["model", "model_repair", "fallback"] = "model"
