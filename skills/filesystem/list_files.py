from __future__ import annotations

from typing import Any

from april_common.path_security import normalize_existing_path
from skills.base import timed_tool
from skills.filesystem.common import current_path_policy, ignored, read_gitignore_patterns
from skills.schemas import ToolDefinition, ToolResult


async def list_files(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        policy = current_path_policy()
        directory = normalize_existing_path(args["path"], policy)
        if not directory.is_dir():
            return ToolResult(
                ok=False,
                stderr="Path is not a directory.",
                risk_level="read_only",
                permission_level=1,
            )
        limit = int(args.get("limit", 100))
        patterns = read_gitignore_patterns(directory)
        files: list[str] = []
        for path in sorted(directory.rglob("*")):
            if ignored(path, root=directory, patterns=patterns):
                continue
            files.append(str(path))
            if len(files) >= limit:
                break
        return ToolResult(
            ok=True,
            stdout="\n".join(files),
            data={"files": files},
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


def list_files_definition() -> ToolDefinition:
    return ToolDefinition(
        name="list_files",
        description="List files under an allowed directory.",
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"coding_agent", "reading_agent"},
        executor=list_files,
        affected_paths=lambda args: [str(args.get("path", ""))],
    )
