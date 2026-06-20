from __future__ import annotations

from typing import Any

from april_common.path_security import ensure_text_file, normalize_existing_path
from skills.base import timed_tool
from skills.filesystem.common import current_path_policy
from skills.schemas import ToolDefinition, ToolResult


async def read_file(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        policy = current_path_policy()
        path = normalize_existing_path(args["path"], policy)
        ensure_text_file(path, max_bytes=policy.max_read_bytes)
        start_line = int(args.get("start_line", 1))
        end_line = args.get("end_line")
        end_line_int = int(end_line) if end_line is not None else None
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start_line - 1 : end_line_int]
        numbered = "\n".join(
            f"{line_number}: {line}" for line_number, line in enumerate(selected, start=start_line)
        )
        return ToolResult(
            ok=True,
            stdout=numbered,
            data={
                "path": str(path),
                "start_line": start_line,
                "end_line": end_line_int or len(lines),
            },
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


def read_file_definition() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read bounded text from an allowed file.",
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"coding_agent", "reading_agent"},
        executor=read_file,
        affected_paths=lambda args: [str(args.get("path", ""))],
    )
