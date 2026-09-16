"""Smoke test: launch the MCP server over stdio and exercise its tools."""

import asyncio
import os
import sys
import tempfile

import pytest

pytest.importorskip("mcp")


async def _smoke():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    env["AGENTMW_HOME"] = tempfile.mkdtemp()

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentmw.cli", "serve"],
        env=env,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert {"agentmw_recall", "agentmw_save", "agentmw_check_trace", "agentmw_stats"}.issubset(names)

            stats = await session.call_tool("agentmw_stats", {})
            assert "patterns" in stats.content[0].text

            saved = await session.call_tool(
                "agentmw_save",
                {"task": "fix retry that drops email", "pattern": "look for .pop('email')"},
            )
            assert "saved_id" in saved.content[0].text

            recalled = await session.call_tool(
                "agentmw_recall",
                {"task": "email disappears on user-creation retry"},
            )
            text = recalled.content[0].text
            assert "pop" in text or "email" in text


def test_mcp_server_stdio_roundtrip():
    asyncio.run(_smoke())
