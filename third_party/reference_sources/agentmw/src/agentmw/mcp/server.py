"""MCP server exposing agentmw to any MCP-compatible client.

Run with:
    agentmw serve            # stdio transport, default
    agentmw serve --http     # http transport on 127.0.0.1:8765

Tools exposed:
    - agentmw_recall(task)              -> JSON list of reasoning patterns
    - agentmw_save(task, pattern, ...)  -> persist a new pattern
    - agentmw_check_trace(trace_json)   -> run monitors on a serialized trace
    - agentmw_stats()                   -> library size, semantic enabled?
"""

from __future__ import annotations

import json
from typing import Any

from agentmw.core.memory import ReasoningLibrary
from agentmw.core.monitors import run_monitors


def build_server(library: ReasoningLibrary | None = None) -> Any:
    """Build a FastMCP server. Imported lazily so the dep is optional."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "agentmw[mcp] is not installed. Run: pip install 'agentmw[mcp]'"
        ) from e

    lib = library or ReasoningLibrary()
    mcp = FastMCP("agentmw")

    @mcp.tool()
    def agentmw_recall(task: str, limit: int = 3) -> str:
        """Recall up to `limit` reasoning patterns similar to `task`."""
        patterns = lib.recall(task, limit=limit)
        return json.dumps(
            [
                {
                    "id": p.id,
                    "outcome": p.outcome,
                    "pattern": p.pattern_text,
                    "score": round(p.score, 4),
                }
                for p in patterns
            ],
            indent=2,
        )

    @mcp.tool()
    def agentmw_save(task: str, pattern: str, outcome: str = "success") -> str:
        """Persist a reasoning pattern. `outcome` is 'success' or 'failure'."""
        pid = lib.save(task, pattern, outcome=outcome)
        return json.dumps({"saved_id": pid, "semantic": lib.semantic_enabled})

    @mcp.tool()
    def agentmw_check_trace(trace_json: str) -> str:
        """Run heuristic monitors on a serialized list of messages (Anthropic format)."""
        try:
            messages = json.loads(trace_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid JSON: {e}"})
        report = run_monitors(messages)
        return json.dumps(
            {
                "triggered": [
                    {"name": r.name, "reason": r.reason, "correction": r.correction}
                    for r in report.triggered
                ],
                "fired_count": len(report.triggered),
            },
            indent=2,
        )

    @mcp.tool()
    def agentmw_stats() -> str:
        """Inspect the reasoning library."""
        return json.dumps(
            {
                "patterns": lib.count(),
                "semantic_enabled": lib.semantic_enabled,
                "db_path": str(lib.db_path),
            }
        )

    return mcp


def run_stdio() -> None:
    server = build_server()
    server.run()  # FastMCP defaults to stdio
