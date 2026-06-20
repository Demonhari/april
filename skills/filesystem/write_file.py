from __future__ import annotations

from typing import Any

from april_common.errors import PermissionDeniedError
from april_common.path_security import normalize_new_path
from skills.base import timed_tool
from skills.filesystem.common import current_path_policy
from skills.schemas import ToolDefinition, ToolResult


async def write_file(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        policy = current_path_policy()
        content = str(args.get("content", ""))
        if len(content.encode("utf-8")) > policy.max_write_bytes:
            raise PermissionDeniedError("Write exceeds configured maximum size.")
        path = normalize_new_path(args["path"], policy)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            ok=True,
            stdout=f"Wrote {path}",
            data={"path": str(path)},
            risk_level="code_write",
            permission_level=3,
        )

    return await timed_tool(run, risk_level="code_write", permission_level=3)


def write_file_definition() -> ToolDefinition:
    return ToolDefinition(
        name="write_file",
        description="Write a file under an allowed root after approval.",
        permission_level=3,
        risk_level="code_write",
        confirmation_required=True,
        allowed_agents={"coding_agent"},
        executor=write_file,
        affected_paths=lambda args: [str(args.get("path", ""))],
    )
