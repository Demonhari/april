from __future__ import annotations

from services.permissions.engine import PermissionEngine
from services.permissions.schemas import PermissionDecision
from skills.registry import ToolRegistry


class ToolPolicy:
    """Compatibility facade for deterministic tool permission checks."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.engine = PermissionEngine(registry)

    def evaluate(
        self,
        *,
        tool: str,
        args: dict[str, object],
        agent: str,
        model_permission_level: int = 0,
        model_risk_level: str = "none",
    ) -> PermissionDecision:
        return self.engine.evaluate(
            tool=tool,
            args=args,
            agent=agent,
            model_permission_level=model_permission_level,
            model_risk_level=model_risk_level,
        )
