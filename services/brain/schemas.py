from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal[
    "none", "read_only", "safe_write", "code_write", "system_action", "external_action"
]


class BrainDecision(BaseModel):
    intent: str
    agent: str
    model_id: str
    tools_needed: list[str] = Field(default_factory=list)
    memory_queries: list[str] = Field(default_factory=list)
    permission_level: int = Field(ge=0, le=5)
    risk_level: RiskLevel
    needs_confirmation: bool
    task_steps: list[str] = Field(default_factory=list, max_length=8)
    decision_summary: str
    routing_method: Literal["model", "model_repair", "fallback"] = "model"
