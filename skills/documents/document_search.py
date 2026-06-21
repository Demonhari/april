from __future__ import annotations

from typing import Any

from april_common.settings import get_settings
from services.memory.vector_memory import VectorMemory
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def document_search(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        vector = VectorMemory(settings.vector_index_path)
        query = str(args.get("query", ""))
        limit = int(args.get("limit", 5))
        results = vector.search(query, limit=limit, source_type="document")
        chunks = [
            {
                "path": result.metadata.get("path"),
                "start_line": result.metadata.get("start_line"),
                "end_line": result.metadata.get("end_line"),
                "score": result.score,
                "content": result.content,
            }
            for result in results
        ]
        return ToolResult(
            ok=True,
            stdout=f"Found {len(chunks)} document chunks.",
            data={"chunks": chunks, "query": query},
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


def document_search_definition() -> ToolDefinition:
    return ToolDefinition(
        name="document_search",
        description="Search indexed local documents in the local vector store (read-only).",
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"reading_agent", "general_agent"},
        executor=document_search,
        affected_paths=lambda args: [],
    )
