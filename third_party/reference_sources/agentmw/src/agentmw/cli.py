"""Command-line entry point for agentmw."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys

from agentmw import __version__
from agentmw.core.config import AgentmwConfig, default_config
from agentmw.core.memory import ReasoningLibrary
from agentmw.core.providers import select_provider


def _apply_overrides(cfg: AgentmwConfig, args: argparse.Namespace) -> AgentmwConfig:
    """Apply CLI flag overrides on top of env + file config."""
    if getattr(args, "provider", None):
        cfg.provider.name = args.provider
    if getattr(args, "model", None):
        cfg.provider.model = args.model
    if getattr(args, "no_llm", False):
        cfg.pipeline.use_llm = False
    return cfg


def _add_provider_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=["auto", "ollama", "openai", "anthropic", "openrouter", "none"],
                   help="LLM provider for monitors / judge. Overrides AGENTMW_PROVIDER.")
    p.add_argument("--model", help="Model name (provider-specific). Overrides AGENTMW_MODEL.")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM monitors entirely; use heuristics only.")


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"agentmw {__version__}")
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(default_config(), args)
    provider = select_provider(cfg.provider)
    payload = {
        "monitors": dataclasses.asdict(cfg.monitors),
        "compression": dataclasses.asdict(cfg.compression),
        "memory": dataclasses.asdict(cfg.memory),
        "provider": {
            **dataclasses.asdict(cfg.provider),
            "api_key": "***" if cfg.provider.api_key else None,
            "resolved_to": provider.name,
            "resolved_model": getattr(provider, "model", None),
            "available": getattr(provider, "available", False),
        },
        "pipeline": dataclasses.asdict(cfg.pipeline),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    cfg = default_config()
    lib = ReasoningLibrary(db_path=cfg.memory.db_path)
    if args.action == "count":
        print(lib.count())
    elif args.action == "save":
        if not args.task or not args.pattern:
            print("--task and --pattern are required for `save`", file=sys.stderr)
            return 2
        pid = lib.save(args.task, args.pattern, outcome=args.outcome)
        print(json.dumps({"saved_id": pid}))
    elif args.action == "recall":
        if not args.task:
            print("--task is required for `recall`", file=sys.stderr)
            return 2
        results = lib.recall(args.task, limit=args.limit)
        print(json.dumps([dataclasses.asdict(p) for p in results], indent=2))
    return 0


def _cmd_timeline(args: argparse.Namespace) -> int:
    from agentmw.core.timeline import build_timeline, load_trace, render_ascii

    cfg = _apply_overrides(default_config(), args)
    messages = load_trace(args.trace)
    memory = None if args.no_memory else ReasoningLibrary(db_path=cfg.memory.db_path)
    provider = select_provider(cfg.provider) if (args.judge and cfg.pipeline.use_llm) else None
    report = build_timeline(
        messages,
        config=cfg,
        provider=provider,
        memory=memory,
        use_judge=args.judge,
    )
    print(render_ascii(report))
    return 0


def _cmd_serve(_args: argparse.Namespace) -> int:
    from agentmw.mcp.server import run_stdio
    run_stdio()
    return 0


def _cmd_stats(_args: argparse.Namespace) -> int:
    from agentmw.core.telemetry import global_telemetry
    print(json.dumps(global_telemetry().to_dict(), indent=2))
    return 0


def _cmd_check_live(args: argparse.Namespace) -> int:
    """PreToolUse hook entry point for Claude Code.

    Reads the hook payload from stdin, appends the tool call to a per-session
    rolling JSONL, runs heuristic monitors on the last N calls, and — if any
    fires — returns a PreToolUse JSON object that injects an advisory note
    Claude will see as additional context. Never blocks the tool.

    Always exits 0 so a hook bug never breaks the user's session.
    """
    from agentmw.core.heuristics import run_heuristics
    from agentmw.core.config import MonitorConfig

    try:
        payload = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except (ValueError, json.JSONDecodeError):
        payload = {}

    tool_name = payload.get("tool_name") or args.tool_name or ""
    tool_input = payload.get("tool_input") or {}
    session_id = payload.get("session_id") or os.environ.get("CLAUDE_SESSION_ID", "default")

    if not tool_name:
        # Nothing to check.
        return 0

    home = os.environ.get("AGENTMW_HOME") or os.path.expanduser("~/.agentmw")
    os.makedirs(os.path.join(home, "hook"), exist_ok=True)
    log_path = os.path.join(home, "hook", f"{session_id}.jsonl")

    if args.reset:
        try:
            os.unlink(log_path)
        except FileNotFoundError:
            pass
        return 0

    # Append this tool_use as a fake assistant message so monitors see it.
    record = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": str(int(_now_ms())),
                     "name": tool_name, "input": tool_input}],
    }
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        return 0

    # Load the last N records as the rolling trace.
    try:
        with open(log_path) as f:
            lines = f.readlines()[-args.window:]
        messages = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError):
        return 0

    results = run_heuristics(messages, MonitorConfig())
    triggered = [r for r in results if r.triggered]
    if not triggered:
        return 0

    warning = "[agentmw] " + " ; ".join(f"{r.name}: {r.reason}" for r in triggered)
    response = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": warning + "\nConsider: " + " | ".join(
                r.correction for r in triggered if r.correction
            ),
        }
    }
    print(json.dumps(response))
    return 0


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)


def _cmd_extract(args: argparse.Namespace) -> int:
    from agentmw.core.extractor import PatternExtractor
    from agentmw.core.sessions import SessionStore

    cfg = _apply_overrides(default_config(), args)
    provider = select_provider(cfg.provider)
    if not getattr(provider, "available", False):
        print(f"provider {provider.name} not available; configure --provider or env vars",
              file=sys.stderr)
        return 2
    memory = ReasoningLibrary(db_path=cfg.memory.db_path)
    store = SessionStore()
    session = store.load(args.id)
    saved = PatternExtractor(provider, memory, dedup_threshold=cfg.extractor.dedup_threshold).extract(
        session.messages, outcome=args.outcome,
    )
    print(json.dumps({"session": session.id,
                      "patterns_saved": [{"task": p.task, "pattern": p.pattern} for p in saved]},
                     indent=2))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    """Save a trace (read from --file or stdin) as a session JSON."""
    from agentmw.core.sessions import Session, SessionStore

    if args.file:
        with open(args.file) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)
    if isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
        task = data.get("task")
    elif isinstance(data, list):
        messages = data
        task = None
    else:
        print("input must be a list of messages or {messages: [...], task?: ...}", file=sys.stderr)
        return 2
    session = Session.from_messages(messages, task=task)
    store = SessionStore()
    path = store.save(session)
    print(json.dumps({"id": session.id, "path": str(path), "messages": len(messages)}))
    return 0


def _cmd_sessions(args: argparse.Namespace) -> int:
    from agentmw.core.sessions import SessionStore

    store = SessionStore()
    if args.action == "list":
        rows = store.list()
        for s in rows[: args.limit]:
            print(f"{s.id}  {len(s.messages):>3} msg  {s.task[:60]}")
        return 0
    if args.action == "show":
        if not args.id:
            print("--id is required", file=sys.stderr)
            return 2
        s = store.load(args.id)
        print(json.dumps({"id": s.id, "task": s.task, "messages": len(s.messages),
                          "created_at": s.created_at}, indent=2))
        return 0
    return 1


def _cmd_replay(args: argparse.Namespace) -> int:
    from agentmw.core.replay import build_counterfactual, render_counterfactual
    from agentmw.core.sessions import SessionStore
    from agentmw.core.timeline import build_timeline, render_ascii

    cfg = _apply_overrides(default_config(), args)
    store = SessionStore()
    session = store.load(args.id)
    memory = None if args.no_memory else ReasoningLibrary(db_path=cfg.memory.db_path)
    provider = select_provider(cfg.provider) if cfg.pipeline.use_llm else None

    report = build_timeline(
        session.messages,
        config=cfg,
        provider=provider,
        memory=memory,
        use_judge=args.judge,
    )
    print(render_ascii(report))

    if args.counterfactual:
        if provider is None or not getattr(provider, "available", False):
            print("\n(no provider available — skipping counterfactual)")
            return 0
        try:
            cf = build_counterfactual(session.messages, report, provider)
        except Exception as e:
            print(f"\n(counterfactual failed: {e})")
            return 0
        if cf is None:
            print("\n(no first-fire step — nothing to counterfact)")
            return 0
        print(render_counterfactual(report, cf))
    return 0


def _cmd_demo(_args: argparse.Namespace) -> int:
    """Mock demo: exercises monitors + compression + memory without an API call."""
    from agentmw import wrap
    from agentmw.wrap import WrapConfig

    class _MockMessages:
        def create(self, **kwargs):
            return {
                "id": "mock_msg_01",
                "received_messages": len(kwargs.get("messages", [])),
                "received_system_len": len(kwargs.get("system") or ""),
            }

    class _MockClient:
        def __init__(self):
            self.messages = _MockMessages()

    messages = [
        {"role": "user", "content": "Find the bug in the auth flow and fix it."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "grep", "input": {"q": "login"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 1500}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Actually, let me reconsider. Wait, no, that's wrong."},
            {"type": "tool_use", "id": "t2", "name": "grep", "input": {"q": "login"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "x" * 1500}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t3", "name": "grep", "input": {"q": "login"}}]},
    ]

    full_cfg = default_config()
    full_cfg.pipeline.use_llm = False  # demo runs offline
    cfg = WrapConfig()
    client = wrap(_MockClient(), config=cfg, agentmw_config=full_cfg)
    client.memory.save(
        "Find the bug in the auth flow and fix it.",
        "Start by checking session token validation, that's where similar bugs hid before.",
        outcome="success",
    )

    result = client.messages.create(messages=messages, system="You are a code agent.")
    trace = cfg.traces[-1]

    print("=" * 60)
    print("agentmw demo")
    print("=" * 60)
    print(f"Pipeline:   provider={trace.monitors.provider_name or 'n/a'} "
          f"llm={trace.monitors.used_llm} heuristics={trace.monitors.used_heuristics}")
    print(f"Monitors fired:        {[r.name for r in trace.monitors.triggered]}")
    for r in trace.monitors.triggered:
        print(f"  - {r.name}: {r.reason}")
    print(f"Compression bytes:     {trace.compression.bytes_before} -> {trace.compression.bytes_after} "
          f"({trace.compression.ratio*100:.1f}% saved, {trace.compression.truncated_blocks} blocks truncated)")
    print(f"Reasoning patterns recalled: {trace.recalled_patterns}")
    print()
    print("System note injected:")
    print("-" * 60)
    print(trace.system_note or "(none)")
    print("-" * 60)
    print(f"Mock response: {result}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(prog="agentmw", description="agentmw CLI")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="-v for INFO, -vv for DEBUG")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)
    sub.add_parser("demo", help="run a self-contained smoke test").set_defaults(func=_cmd_demo)
    sub.add_parser("serve", help="run the MCP server (stdio transport)").set_defaults(func=_cmd_serve)
    sub.add_parser("stats", help="print telemetry counters").set_defaults(func=_cmd_stats)

    chk = sub.add_parser("check-live",
                         help="PreToolUse hook: append tool call to rolling log, warn on loop/redundancy")
    chk.add_argument("--window", type=int, default=12, help="how many recent calls to consider")
    chk.add_argument("--reset", action="store_true", help="clear the rolling log for this session")
    chk.add_argument("--tool-name", help="override tool name (for testing without stdin payload)")
    chk.set_defaults(func=_cmd_check_live)

    ext = sub.add_parser("extract", help="extract reusable patterns from a saved session")
    ext.add_argument("id", help="session id")
    ext.add_argument("--outcome", default="success", choices=["success", "failure"])
    _add_provider_flags(ext)
    ext.set_defaults(func=_cmd_extract)

    rec = sub.add_parser("record", help="save a trace (stdin or --file) as a session")
    rec.add_argument("--file", help="read trace JSON from this file instead of stdin")
    rec.set_defaults(func=_cmd_record)

    ses = sub.add_parser("sessions", help="list / inspect saved sessions")
    ses.add_argument("action", choices=["list", "show"])
    ses.add_argument("--id", help="session id (prefix accepted)")
    ses.add_argument("--limit", type=int, default=20)
    ses.set_defaults(func=_cmd_sessions)

    rep = sub.add_parser("replay", help="replay a saved session with timeline + counterfactual")
    rep.add_argument("id", help="session id (prefix accepted)")
    rep.add_argument("--judge", action="store_true", help="LLM judge on each assistant turn")
    rep.add_argument("--counterfactual", action="store_true",
                     help="ask the provider to simulate the divergent branch from first-fire")
    rep.add_argument("--no-memory", action="store_true")
    _add_provider_flags(rep)
    rep.set_defaults(func=_cmd_replay)

    cfg = sub.add_parser("config", help="inspect resolved configuration")
    cfg_sub = cfg.add_subparsers(dest="action", required=True)
    cfg_show = cfg_sub.add_parser("show", help="print effective config (env + file + defaults)")
    _add_provider_flags(cfg_show)
    cfg_show.set_defaults(func=_cmd_config_show)

    tl = sub.add_parser("timeline", help="time-travel a saved trace; show first-fire and waste")
    tl.add_argument("trace", help="path to a trace JSON file")
    tl.add_argument("--judge", action="store_true", help="run the LLM judge on each assistant turn")
    tl.add_argument("--no-memory", action="store_true", help="skip reasoning-library recall")
    _add_provider_flags(tl)
    tl.set_defaults(func=_cmd_timeline)

    mem = sub.add_parser("memory", help="inspect or manipulate the reasoning library")
    mem.add_argument("action", choices=["count", "save", "recall"])
    mem.add_argument("--task")
    mem.add_argument("--pattern")
    mem.add_argument("--outcome", default="success")
    mem.add_argument("--limit", type=int, default=3)
    mem.set_defaults(func=_cmd_memory)

    args = parser.parse_args(argv)
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
