from __future__ import annotations

from typing import Any

from april_common.path_security import ensure_text_file, normalize_existing_path
from skills.base import timed_tool
from skills.filesystem.common import (
    current_path_policy,
    ignored,
    read_gitignore_patterns,
    safe_regex,
)
from skills.schemas import ToolDefinition, ToolResult


async def search_files(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        policy = current_path_policy()
        root = normalize_existing_path(args["path"], policy)
        pattern = str(args["query"])
        regex = safe_regex(pattern) if args.get("regex", False) else None
        limit = int(args.get("limit", 50))
        patterns = read_gitignore_patterns(root)
        matches: list[dict[str, object]] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or ignored(file_path, root=root, patterns=patterns):
                continue
            try:
                ensure_text_file(file_path, max_bytes=policy.max_read_bytes)
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for line_number, line in enumerate(lines, start=1):
                found = regex.search(line) if regex else pattern.lower() in line.lower()
                if found:
                    matches.append(
                        {
                            "path": str(file_path),
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= limit:
                        stdout = "\n".join(
                            f"{match['path']}:{match['line']}: {match['text']}" for match in matches
                        )
                        return ToolResult(
                            ok=True,
                            stdout=stdout,
                            data={"matches": matches},
                            risk_level="read_only",
                            permission_level=1,
                        )
        stdout = "\n".join(f"{match['path']}:{match['line']}: {match['text']}" for match in matches)
        return ToolResult(
            ok=True,
            stdout=stdout,
            data={"matches": matches},
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


def search_files_definition() -> ToolDefinition:
    return ToolDefinition(
        name="search_files",
        description="Search text files under an allowed root.",
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"coding_agent", "reading_agent", "reasoning_agent"},
        executor=search_files,
        affected_paths=lambda args: [str(args.get("path", ""))],
    )
