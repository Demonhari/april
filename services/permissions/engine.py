from __future__ import annotations

from typing import Any

from april_common.errors import PermissionDeniedError
from services.permissions.risk import level_for_risk, max_risk
from services.permissions.schemas import PermissionDecision
from skills.registry import ToolRegistry


class PermissionEngine:
    def __init__(self, registry: ToolRegistry, *, approval_required_at: int = 3) -> None:
        self.registry = registry
        self.approval_required_at = approval_required_at

    def evaluate(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        agent: str,
        model_permission_level: int = 0,
        model_risk_level: str = "none",
    ) -> PermissionDecision:
        definition = self.registry.get(tool)
        if definition is None:
            raise PermissionDeniedError("Unknown tool is denied.", {"tool": tool})
        if agent not in definition.allowed_agents and "*" not in definition.allowed_agents:
            raise PermissionDeniedError(
                "Tool is not allowed for this agent.",
                {"tool": tool, "agent": agent},
            )
        arg_risk = definition.argument_risk(args)
        risk = max_risk(model_risk_level, definition.risk_level, arg_risk)
        permission_level = max(
            model_permission_level, definition.permission_level, level_for_risk(risk)
        )
        confirmation_required = (
            definition.confirmation_required or permission_level >= self.approval_required_at
        )
        affected_paths = definition.affected_paths(args)
        return PermissionDecision(
            allowed=True,
            permission_level=permission_level,
            risk_level=risk,
            confirmation_required=confirmation_required,
            reason="Allowed by deterministic tool policy.",
            affected_paths=affected_paths,
        )
