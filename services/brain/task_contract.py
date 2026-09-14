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


_TASK_TYPE_ORDER: dict[TaskType, int] = {
    "explanation": 0,
    "investigation": 1,
    "code_modification": 2,
    "verified_code_modification": 3,
}


def task_contract_for_request(
    *,
    run_id: str,
    user_goal: str,
    agent_name: str,
    agent_tools: set[str],
    project_id: str | None,
    project_root: str | None,
    intent: str,
    permission_level: int,
    risk_level: str,
    allowed_scope: tuple[str, ...] = (),
) -> TaskContract:
    """Build the narrow trusted contract from routed application state.

    This function deliberately does not inspect model-generated prose. The
    router's normalized intent, the configured agent, and the selected project
    are the authority inputs.
    """

    write_tools = {
        "write_file",
        "patch_applier",
        "git_commit",
        "run_command",
        "test_runner",
    }
    investigation_intents = {"coding_repo_analysis", "document_reading"}
    modification_intents = {"code_modification", "patch_proposal"}
    has_write_capability = bool(agent_tools & write_tools)
    if intent in modification_intents and project_id is not None:
        task_type: TaskType = "verified_code_modification"
    elif agent_name == "coding_agent" and has_write_capability and project_id is not None:
        task_type = (
            "verified_code_modification" if intent not in investigation_intents else "investigation"
        )
    elif intent in investigation_intents:
        task_type = "investigation"
    else:
        task_type = "explanation"
    verified = task_type == "verified_code_modification"
    return TaskContract(
        task_id=run_id,
        run_id=run_id,
        user_goal=user_goal,
        project_id=project_id,
        project_root=project_root,
        task_type=task_type,
        allowed_scope=allowed_scope,
        risk_class=(
            "approval_required"
            if permission_level >= 3
            else ("write" if has_write_capability else "read_only")
        ),
        success_criteria=("current machine verification passes",) if verified else (),
        verification=VerificationRequirements(
            required=verified,
            commands=("test_runner",) if verified else (),
            require_current_repository_state=verified,
        ),
        maximum_specialist_depth=1 if verified else 0,
        maximum_replan_attempts=1 if verified else 0,
        capability_ceiling=CapabilityCeiling(
            allowed_tools=frozenset(agent_tools),
            maximum_risk_level=max(0, min(permission_level, 4)),
            project_id=project_id,
            project_root=project_root,
        ),
    )


def constrain_task_contract(
    derived: TaskContract,
    requested: TaskContract,
    *,
    request_id: str,
) -> TaskContract:
    """Accept a caller contract only when it is a narrowing of trusted state."""

    if requested.run_id != request_id:
        raise ValueError("Task contract run identity does not match request.")
    if requested.project_id not in {None, derived.project_id}:
        raise ValueError("Task contract project exceeds the selected project.")
    if requested.project_root not in {None, derived.project_root}:
        raise ValueError("Task contract root exceeds the selected project.")
    if not requested.capability_ceiling.allowed_tools <= derived.capability_ceiling.allowed_tools:
        raise ValueError("Task contract tools exceed the trusted agent ceiling.")
    if (
        requested.capability_ceiling.maximum_risk_level
        > derived.capability_ceiling.maximum_risk_level
    ):
        raise ValueError("Task contract risk exceeds the trusted agent ceiling.")
    if _TASK_TYPE_ORDER[requested.task_type] > _TASK_TYPE_ORDER[derived.task_type]:
        raise ValueError("Task contract task type exceeds the trusted route.")
    if derived.verification.required and not requested.verification.required:
        raise ValueError("Task contract cannot remove required verification.")
    if requested.maximum_specialist_depth > derived.maximum_specialist_depth:
        raise ValueError("Task contract delegation depth exceeds the trusted ceiling.")
    if requested.maximum_replan_attempts > derived.maximum_replan_attempts:
        raise ValueError("Task contract re-plan budget exceeds the trusted ceiling.")
    return requested.model_copy(
        update={
            "task_id": derived.task_id,
            "run_id": request_id,
            "project_id": derived.project_id,
            "project_root": derived.project_root,
            "allowed_scope": derived.allowed_scope,
            "capability_ceiling": CapabilityCeiling(
                allowed_tools=requested.capability_ceiling.allowed_tools,
                maximum_risk_level=requested.capability_ceiling.maximum_risk_level,
                project_id=derived.project_id,
                project_root=derived.project_root,
            ),
        }
    )
