from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ToolExecutor = Callable[[dict[str, Any]], Awaitable["ToolResult"]]
ArgumentRiskEvaluator = Callable[[dict[str, Any]], str]
AffectedPathResolver = Callable[[dict[str, Any]], list[str]]


class ToolResult(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    risk_level: str
    permission_level: int
    duration_ms: int = 0


class ToolDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    permission_level: int
    risk_level: str
    confirmation_required: bool = False
    allowed_agents: set[str] = Field(default_factory=set)
    timeout_seconds: float = 15.0
    executor: ToolExecutor = Field(exclude=True)
    argument_risk: ArgumentRiskEvaluator = Field(
        default=lambda args: "none",
        exclude=True,
    )
    affected_paths: AffectedPathResolver = Field(default=lambda args: [], exclude=True)


class ToolRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
