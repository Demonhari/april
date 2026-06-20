from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal[
    "none", "read_only", "safe_write", "code_write", "system_action", "external_action"
]

RISK_ORDER: dict[str, int] = {
    "none": 0,
    "read_only": 1,
    "safe_write": 2,
    "code_write": 3,
    "system_action": 4,
    "external_action": 5,
}


class PermissionDecision(BaseModel):
    allowed: bool
    permission_level: int
    risk_level: RiskLevel
    confirmation_required: bool
    reason: str
    affected_paths: list[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    tool: str
    args: dict[str, Any]
    permission_level: int
    risk_level: RiskLevel
    affected_paths: list[str] = Field(default_factory=list)
    expected_side_effects: list[str] = Field(default_factory=list)


class ApprovalResponse(BaseModel):
    approval_id: str
    tool: str
    args: dict[str, Any]
    permission_level: int
    risk_level: RiskLevel
    affected_paths: list[str]
    expected_side_effects: list[str]
    expires_at: str


class ApprovalExecutionResult(BaseModel):
    approval_id: str
    ok: bool
    message: str
    result: dict[str, Any] = Field(default_factory=dict)
