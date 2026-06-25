from __future__ import annotations

from typing import Any

from april_common.errors import AprilError
from services.permissions.cleanup import build_cleanup_manifest
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def plan_log_cleanup(args: dict[str, Any]) -> ToolResult:
    """Level 1, read-only: enumerate deletable files into an immutable manifest.

    Deletes nothing. Accepts only a controlled ``target`` enum and a bounded
    ``older_than_days``; the root is derived from settings, never from the caller.
    """

    async def run() -> ToolResult:
        target = str(args.get("target", "logs"))
        try:
            older_than_days = int(args.get("older_than_days", 0))
        except (TypeError, ValueError):
            return ToolResult(
                ok=False,
                stderr="older_than_days must be an integer.",
                risk_level="read_only",
                permission_level=1,
            )
        try:
            result = build_cleanup_manifest(target=target, older_than_days=older_than_days)
        except AprilError as exc:
            return ToolResult(
                ok=False,
                stderr=exc.message,
                data=exc.details,
                risk_level="read_only",
                permission_level=1,
            )
        return ToolResult(
            ok=True,
            stdout=(
                f"{result['candidate_count']} file(s), {result['total_bytes']} bytes under "
                f"{result['target']}. Approve apply_log_cleanup with this manifest to delete."
            ),
            data=result,
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


async def apply_log_cleanup(args: dict[str, Any]) -> ToolResult:
    """Guard executor.

    ``apply_log_cleanup`` is Level 4 and only ever runs through the approved
    execution path (``apply_approved_log_cleanup``), which is bound to the
    immutable manifest. Reaching this executor directly means the approval
    boundary was bypassed, so it fails closed.
    """
    return ToolResult(
        ok=False,
        stderr="apply_log_cleanup requires an exact, approved cleanup manifest.",
        risk_level="system_action",
        permission_level=4,
    )


def plan_log_cleanup_definition() -> ToolDefinition:
    return ToolDefinition(
        name="plan_log_cleanup",
        description=(
            "Plan a scoped cleanup of old files under an APRIL-owned root "
            "(logs or audio_cache). Read-only; produces an immutable manifest."
        ),
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"system_action_agent"},
        executor=plan_log_cleanup,
        affected_paths=lambda args: [],
    )


def apply_log_cleanup_definition() -> ToolDefinition:
    return ToolDefinition(
        name="apply_log_cleanup",
        description=(
            "Delete exactly the files in a previously planned cleanup manifest. "
            "Level 4 system action; requires exact one-time approval."
        ),
        permission_level=4,
        risk_level="system_action",
        confirmation_required=True,
        allowed_agents={"system_action_agent"},
        executor=apply_log_cleanup,
        affected_paths=lambda args: [],
    )
