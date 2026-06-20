from __future__ import annotations

from pathlib import Path
from typing import Any

from april_common.settings import get_settings
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def search_notes(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        notes_dir = settings.resolve_path(Path("data/notes"))
        query = str(args["query"]).lower()
        matches: list[dict[str, str]] = []
        if notes_dir.exists():
            for path in sorted(notes_dir.glob("*.md")):
                text = path.read_text(encoding="utf-8", errors="replace")
                if query in text.lower():
                    matches.append({"path": str(path), "preview": text[:300]})
        return ToolResult(
            ok=True,
            stdout="\n".join(match["path"] for match in matches),
            data={"matches": matches},
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


def search_notes_definition() -> ToolDefinition:
    return ToolDefinition(
        name="search_notes",
        description="Search local APRIL notes.",
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"general_agent", "creative_agent"},
        executor=search_notes,
    )
