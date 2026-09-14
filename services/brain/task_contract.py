"""Small, typed authority contract for one APRIL run.

The model may describe work against this contract, but it cannot widen any of
the contract's scope, risk, or verification requirements.
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

TaskType = Literal[
    "explanation", "investigation", "code_modification", "verified_code_modification"
]
RiskClass = Literal["read_only", "write", "approval_required"]


class CapabilityCeiling(BaseModel):
    allowed_tools: frozenset[str] = frozenset()
    maximum_risk_level: int = Field(default=1, ge=0, le=4)
    project_id: str | None = None
    project_root: str | None = None


class VerificationRequirements(BaseModel):
    required: bool = False
    commands: tuple[str, ...] = ()
    require_current_repository_state: bool = True


class TaskContract(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    user_goal: str = Field(min_length=1, max_length=16_384)
    project_id: str | None = None
    project_root: str | None = None
    task_type: TaskType = "explanation"
    allowed_scope: tuple[str, ...] = ()
    risk_class: RiskClass = "read_only"
    success_criteria: tuple[str, ...] = ()
    verification: VerificationRequirements = Field(default_factory=VerificationRequirements)
    maximum_specialist_depth: int = Field(default=1, ge=0, le=2)
    maximum_replan_attempts: int = Field(default=1, ge=0, le=5)
    capability_ceiling: CapabilityCeiling = Field(default_factory=CapabilityCeiling)

    def can_finish_without_current_verification(self) -> bool:
        return not (
            self.verification.required and self.verification.require_current_repository_state
        )
