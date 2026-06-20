from __future__ import annotations

from typing import Any

from services.permissions.artifacts import store_patch_artifact
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def patch_generator(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        content = str(args.get("patch", ""))
        stored = store_patch_artifact(content.encode("utf-8"))
        return ToolResult(
            ok=True,
            stdout=str(stored["artifact_id"]),
            data={
                "artifact_id": stored["artifact_id"],
                "patch_path": stored["path"],
                "patch_sha256": stored["artifact_id"],
                "patch_byte_length": stored["byte_length"],
            },
            risk_level="safe_write",
            permission_level=2,
        )

    return await timed_tool(run, risk_level="safe_write", permission_level=2)


def patch_generator_definition() -> ToolDefinition:
    return ToolDefinition(
        name="patch_generator",
        description="Write a draft patch file without applying it.",
        permission_level=2,
        risk_level="safe_write",
        allowed_agents={"coding_agent"},
        executor=patch_generator,
        affected_paths=lambda args: [],
    )
