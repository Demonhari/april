"""Praxos MCP server built with mcp-use.

The mcp-use server framework keeps Praxos compatible with MCP clients while
giving us a cleaner developer experience than maintaining protocol plumbing by
hand.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import warnings
from pathlib import Path
from typing import Any

from praxos.ledger import ExperienceLedger


def _prepare_mcp_use_runtime() -> None:
    os.environ.setdefault("MCP_USE_ANONYMIZED_TELEMETRY", "false")
    warnings.filterwarnings(
        "ignore",
        message=r"The default value of `allowed_objects` will change.*",
        category=Warning,
    )


def _ledger(db_path: str | Path | None = None) -> ExperienceLedger:
    return ExperienceLedger(db_path)


def create_mcp_server(db_path: str | Path | None = None, workspace_id: str = "default"):
    """Create a Praxos MCP server using mcp-use."""
    _prepare_mcp_use_runtime()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            from mcp_use.server import MCPServer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError(
            "mcp-use is required for Praxos MCP. Install with `pip install -e .` "
            "or `pip install mcp-use`."
        ) from exc

    ledger = _ledger(db_path)
    server = MCPServer(
        name="Praxos",
        version="0.1.0",
        instructions=(
            "Praxos is the experience OS for AI employees. Use these tools before "
            "and after agent actions to retrieve experience, check risk, record "
            "outcomes, and learn from human feedback."
        ),
    )

    @server.tool()
    def get_experience(
        task: str,
        action: str = "",
        account_id: str = "",
        limit: int = 5,
        workspace: str = workspace_id,
    ) -> dict:
        """Retrieve relevant lessons, policies, evidence, and business context."""
        return ledger.get_experience(
            workspace_id=workspace,
            account_id=account_id,
            task=task,
            action=action,
            limit=limit,
        )

    @server.tool()
    def check_action(
        task: str,
        action: str,
        account_id: str = "",
        workspace: str = workspace_id,
    ) -> dict:
        """Pre-flight an agent action and return allow, warn, or block with reasons."""
        return ledger.check_action(
            workspace_id=workspace,
            account_id=account_id,
            task=task,
            action=action,
        ).to_dict()

    @server.tool()
    def record_outcome(
        task: str,
        action: str,
        agent_id: str = "agent",
        outcome: str = "unknown",
        result: str = "",
        feedback: str = "",
        account_id: str = "",
        customer_id: str = "",
        source_refs: list[str] | None = None,
        evidence: list[dict[str, Any]] | None = None,
        workspace: str = workspace_id,
    ) -> dict:
        """Record an agent episode and optionally compile a lesson from feedback."""
        return ledger.record_episode(
            workspace_id=workspace,
            account_id=account_id,
            customer_id=customer_id,
            agent_id=agent_id,
            task=task,
            action=action,
            outcome=outcome,  # type: ignore[arg-type]
            result=result,
            human_feedback=feedback,
            source_refs=source_refs or [],
            evidence=evidence or [],
        ).to_dict()

    @server.tool()
    def learn_from_feedback(
        agent_id: str,
        task: str,
        action: str,
        feedback: str,
        result: str = "",
        account_id: str = "",
        customer_id: str = "",
        source_refs: list[str] | None = None,
        workspace: str = workspace_id,
    ) -> dict:
        """Create a reusable lesson from human feedback on an agent action."""
        return ledger.learn_from_feedback(
            workspace_id=workspace,
            account_id=account_id,
            customer_id=customer_id,
            agent_id=agent_id,
            task=task,
            action=action,
            feedback=feedback,
            result=result,
            source_refs=source_refs or [],
        ).to_dict()

    return server


def run_mcp_server(
    *,
    db_path: str | Path | None = None,
    workspace_id: str = "default",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8766,
    debug: bool = False,
) -> None:
    server = create_mcp_server(db_path=db_path, workspace_id=workspace_id)
    kwargs: dict[str, Any] = {"transport": transport}
    if transport != "stdio":
        kwargs.update({"host": host, "port": int(port)})
    if debug:
        kwargs["debug"] = True
    server.run(**kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="praxos-mcp")
    parser.add_argument("--db", default=os.getenv("PRAXOS_DB", ""))
    parser.add_argument("--workspace", default=os.getenv("PRAXOS_WORKSPACE", "default"))
    parser.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    run_mcp_server(
        db_path=Path(args.db).expanduser() if args.db else None,
        workspace_id=args.workspace,
        transport=args.transport,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
