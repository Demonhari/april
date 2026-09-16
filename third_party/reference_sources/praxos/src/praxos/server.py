"""HTTP tool server for Praxos agents.

This is intentionally dependency-free. It exposes Praxos as JSON tools and also
supports a minimal MCP-style JSON-RPC surface for local agent runtimes:

- POST /tools/check_action
- POST /tools/get_experience
- POST /tools/record_outcome
- POST /tools/learn_from_feedback
- POST /mcp with methods tools/list and tools/call
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from praxos.ledger import ExperienceLedger


TOOL_DESCRIPTIONS = {
    "get_experience": {
        "description": "Retrieve relevant lessons, policies, evidence, and business context before an agent acts.",
        "required": ["task"],
    },
    "check_action": {
        "description": "Pre-flight an agent action and return allow, warn, or block with reasons.",
        "required": ["task", "action"],
    },
    "record_outcome": {
        "description": "Record an agent episode or update an existing episode with outcome and feedback.",
        "required": [],
    },
    "learn_from_feedback": {
        "description": "Create a reusable lesson from human feedback on an agent action.",
        "required": ["agent_id", "task", "action", "feedback"],
    },
}


def _tool_get_experience(ledger: ExperienceLedger, workspace_id: str, args: dict[str, Any]) -> dict:
    return ledger.get_experience(
        workspace_id=args.get("workspace_id") or workspace_id,
        account_id=args.get("account_id", ""),
        task=args["task"],
        action=args.get("action", ""),
        limit=int(args.get("limit", 5)),
    )


def _tool_check_action(ledger: ExperienceLedger, workspace_id: str, args: dict[str, Any]) -> dict:
    return ledger.check_action(
        workspace_id=args.get("workspace_id") or workspace_id,
        account_id=args.get("account_id", ""),
        task=args["task"],
        action=args["action"],
    ).to_dict()


def _tool_record_outcome(ledger: ExperienceLedger, workspace_id: str, args: dict[str, Any]) -> dict:
    if args.get("episode_id"):
        episode = ledger.record_outcome(
            args["episode_id"],
            outcome=args.get("outcome", "unknown"),
            result=args.get("result", ""),
            human_feedback=args.get("feedback") or args.get("human_feedback", ""),
        )
    else:
        episode = ledger.record_episode(
            workspace_id=args.get("workspace_id") or workspace_id,
            account_id=args.get("account_id", ""),
            customer_id=args.get("customer_id", ""),
            agent_id=args.get("agent_id", "agent"),
            task=args["task"],
            action=args["action"],
            outcome=args.get("outcome", "unknown"),
            result=args.get("result", ""),
            human_feedback=args.get("feedback") or args.get("human_feedback", ""),
            source_refs=args.get("source_refs", []),
            evidence=args.get("evidence", []),
        )
    return episode.to_dict()


def _tool_learn_from_feedback(ledger: ExperienceLedger, workspace_id: str, args: dict[str, Any]) -> dict:
    return ledger.learn_from_feedback(
        workspace_id=args.get("workspace_id") or workspace_id,
        account_id=args.get("account_id", ""),
        customer_id=args.get("customer_id", ""),
        agent_id=args["agent_id"],
        task=args["task"],
        action=args["action"],
        feedback=args["feedback"],
        result=args.get("result", ""),
        source_refs=args.get("source_refs", []),
    ).to_dict()


TOOLS: dict[str, Callable[[ExperienceLedger, str, dict[str, Any]], dict]] = {
    "get_experience": _tool_get_experience,
    "check_action": _tool_check_action,
    "record_outcome": _tool_record_outcome,
    "learn_from_feedback": _tool_learn_from_feedback,
}


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def create_handler(db_path: str | Path | None = None, workspace_id: str = "default"):
    ledger = ExperienceLedger(db_path)

    class PraxosHandler(BaseHTTPRequestHandler):
        server_version = "Praxos/0.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/healthz":
                _json_response(self, 200, {"status": "ok", "service": "praxos"})
                return
            if path in {"/tools", "/mcp/list_tools"}:
                _json_response(self, 200, {"tools": TOOL_DESCRIPTIONS})
                return
            if path == "/stats":
                _json_response(self, 200, ledger.stats(workspace_id=workspace_id))
                return
            _json_response(self, 404, {"error": "not_found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
                if path.startswith("/tools/"):
                    name = path.rsplit("/", 1)[-1]
                    if name not in TOOLS:
                        _json_response(self, 404, {"error": f"unknown tool: {name}"})
                        return
                    result = TOOLS[name](ledger, workspace_id, payload)
                    _json_response(self, 200, {"result": result})
                    return

                if path == "/mcp":
                    response = self._handle_mcp(payload)
                    _json_response(self, 200, response)
                    return

                _json_response(self, 404, {"error": "not_found"})
            except KeyError as exc:
                _json_response(self, 400, {"error": f"missing field: {exc}"})
            except Exception as exc:
                _json_response(self, 400, {"error": str(exc)})

        def _handle_mcp(self, payload: dict) -> dict:
            request_id = payload.get("id")
            method = payload.get("method")
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOL_DESCRIPTIONS}}
            if method == "tools/call":
                params = payload.get("params") or {}
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if name not in TOOLS:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"unknown tool: {name}"},
                    }
                result = TOOLS[name](ledger, workspace_id, arguments)
                return {"jsonrpc": "2.0", "id": request_id, "result": {"content": result}}
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown method: {method}"},
            }

    return PraxosHandler


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: str | Path | None = None,
    workspace_id: str = "default",
) -> None:
    handler = create_handler(db_path=db_path, workspace_id=workspace_id)
    server = ThreadingHTTPServer((host, int(port)), handler)
    print(f"Praxos tool server listening on http://{host}:{port}")
    print("Tools: GET /tools, POST /tools/check_action, POST /mcp")
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="praxos-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default=None)
    parser.add_argument("--workspace", default="default")
    args = parser.parse_args(argv)
    run_server(
        host=args.host,
        port=args.port,
        db_path=Path(args.db).expanduser() if args.db else None,
        workspace_id=args.workspace,
    )


if __name__ == "__main__":
    main()
