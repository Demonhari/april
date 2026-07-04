from __future__ import annotations

from typing import Any

from april_common.errors import PermissionDeniedError
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.risk import level_for_risk
from services.permissions.schemas import ApprovalRequest
from skills.playbooks.loader import PlaybookLoader
from skills.playbooks.schema import PlaybookDefinition
from skills.registry import ToolRegistry

ADOPTION_TOOL = "playbook_adopt"


class PlaybookAdoptionService:
    """Adoption gate for playbooks.

    A playbook's required permission level is derived deterministically from
    the registered tools its steps call (unknown tools are denied outright).
    Playbooks at or above the approval threshold (Level 3+) cannot become
    active without an exact-action approval that binds the full playbook
    definition; approving anything else, or a modified definition, fails.
    Adoption never weakens runtime policy: every L3+ step still raises its own
    exact-action approval when the playbook runs.
    """

    def __init__(
        self,
        *,
        loader: PlaybookLoader,
        tool_registry: ToolRegistry,
        approvals: ApprovalStore,
        memory: SqliteMemory | None = None,
        approval_required_at: int = 3,
    ) -> None:
        self.loader = loader
        self.tool_registry = tool_registry
        self.approvals = approvals
        self.memory = memory
        self.approval_required_at = approval_required_at

    def required_permission_level(self, playbook: PlaybookDefinition) -> int:
        level = 1
        for step in playbook.steps:
            definition = self.tool_registry.get(step.tool)
            if definition is None:
                raise PermissionDeniedError(
                    "Playbook references an unknown tool and cannot be adopted.",
                    {"tool": step.tool, "playbook": playbook.id},
                )
            level = max(
                level,
                definition.permission_level,
                level_for_risk(definition.risk_level),
            )
        return level

    async def adopt(
        self,
        playbook: PlaybookDefinition,
        *,
        actor: str,
        request_id: str,
        approval_id: str | None = None,
    ) -> dict[str, Any]:
        level = self.required_permission_level(playbook)
        args = self._approval_args(playbook)
        if level >= self.approval_required_at:
            if approval_id is None:
                response = await self.approvals.create(
                    ApprovalRequest(
                        tool=ADOPTION_TOOL,
                        args=args,
                        agent="local-user",
                        permission_level=level,
                        risk_level="system_action",
                        expected_side_effects=[
                            "Activate a playbook whose steps can request "
                            f"Level {level} actions (each still needs its own "
                            "exact-action approval at run time)."
                        ],
                    ),
                    actor=actor,
                    request_id=request_id,
                )
                return {
                    "status": "pending_approval",
                    "required_permission_level": level,
                    "approval": response.model_dump(),
                }
            # Exact-action approval: the canonical hash binds this precise
            # playbook definition; any modification invalidates the approval.
            await self.approvals.approve_exact(
                approval_id=approval_id,
                tool=ADOPTION_TOOL,
                args=args,
                actor=actor,
                request_id=request_id,
            )
        path = self.loader.adopt(playbook)
        if self.memory is not None:
            await self.memory.upsert_playbook(
                playbook_id=playbook.id,
                name=playbook.name,
                source="adopted",
                status="active",
                trigger_examples=list(playbook.trigger_examples),
                steps=[step.model_dump() for step in playbook.steps],
                required_permission_level=level,
            )
        if level >= self.approval_required_at and approval_id is not None:
            await self.approvals.consume(
                approval_id=approval_id,
                result={"adopted": True, "playbook_id": playbook.id, "path": str(path)},
                actor=actor,
                request_id=request_id,
            )
        return {
            "status": "adopted",
            "adopted": True,
            "id": playbook.id,
            "path": str(path),
            "required_permission_level": level,
        }

    @staticmethod
    def _approval_args(playbook: PlaybookDefinition) -> dict[str, Any]:
        return {"playbook": playbook.model_dump(mode="json")}
