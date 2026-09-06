from __future__ import annotations

# Command modules load after Typer groups exist; public re-exports are intentional.
# ruff: noqa: E402, F401
import asyncio
import sys
import uuid
from collections.abc import Coroutine
from typing import Any

import typer
from rich.prompt import Prompt

from apps.cli.client import ApiOfflineError, ApiResponseError, AprilApiClient
from apps.cli.groups import (
    agent_app,
    app,
    conversation_app,
    daemon_app,
    doc_app,
    evolve_app,
    jobs_app,
    memory_app,
    model_app,
    playbook_app,
    project_app,
    reminder_app,
    task_app,
    voice_app,
)
from apps.cli.render import (
    console,
    print_approvals,
    print_briefing,
    print_jsonish,
    print_models,
)
from april_common.audit import AuditStartupBlocked, audit_startup_decision
from april_common.settings import get_settings

app.add_typer(model_app, name="model")
app.add_typer(project_app, name="project")
app.add_typer(memory_app, name="memory")
app.add_typer(voice_app, name="voice")
app.add_typer(conversation_app, name="conversation")
app.add_typer(agent_app, name="agent")
app.add_typer(reminder_app, name="reminder")
app.add_typer(task_app, name="task")
app.add_typer(doc_app, name="doc")
app.add_typer(daemon_app, name="daemon")
app.add_typer(playbook_app, name="playbook")
app.add_typer(evolve_app, name="evolve")
app.add_typer(jobs_app, name="jobs")

_DAEMON_AUTOSTART_REPORTED = False
_CHAT_MODES = {"standard", "deep", "council"}


def client() -> AprilApiClient:
    settings = get_settings()
    return AprilApiClient(
        f"http://{settings.api.host}:{settings.api.port}",
        settings.api.token or "",
        timeout=settings.runtime.request_timeout_seconds,
    )


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        return asyncio.run(coro)
    except ApiOfflineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _maybe_autostart_daemon() -> None:
    """Best-effort apriald autostart before attach/one-shot, when configured."""
    global _DAEMON_AUTOSTART_REPORTED
    settings = get_settings()
    decision = audit_startup_decision(settings)
    if not decision.accepted:
        console.print(decision.operator_message, markup=False)
        raise typer.Exit(1)
    if not settings.daemon.autostart_on_cli:
        return
    try:
        # Runner-managed services are already the authenticated installation
        # lifecycle.  Do not start a second supervisor merely because its own
        # PID file is absent; a second supervisor would restart services after
        # an explicit runner stop.
        from apps.runner.service_manager import AprilServiceManager

        existing = AprilServiceManager(home=settings.home).status()
        if existing.runtime.running or existing.api.running:
            return
        from apps.daemon.apriald import autostart_if_needed
    except ImportError:
        if not _DAEMON_AUTOSTART_REPORTED:
            console.print(
                "[yellow]Daemon autostart unavailable: apps.daemon.apriald is not "
                "implemented yet.[/yellow]"
            )
            _DAEMON_AUTOSTART_REPORTED = True
        return
    try:
        autostart_if_needed(settings, fake_backend=settings.runtime.backend == "fake")
    except AuditStartupBlocked as exc:
        console.print(exc.decision.operator_message, markup=False)
        raise typer.Exit(1) from exc
    except Exception as exc:
        if not _DAEMON_AUTOSTART_REPORTED:
            console.print(f"[yellow]Daemon autostart failed: {exc}[/yellow]")
            _DAEMON_AUTOSTART_REPORTED = True
        # Autostart is a convenience; attach/one-shot still report a clear
        # offline error if the API stays unreachable.
        return


def _print_chat_result(result: dict[str, Any]) -> None:
    console.print(result["final_message"])
    if result.get("pending_approval"):
        console.print("[yellow]Approval required:[/yellow]")
        print_jsonish(result["pending_approval"])


def _handle_repl_command(message: str, conversation_id: str | None) -> bool:
    stripped = message.strip()
    if not stripped.startswith("/"):
        return False
    command, _, rest = stripped.partition(" ")
    if command == "/status":
        health = run(client().get("/health", auth=False))
        sessions_data = run(client().get("/sessions"))
        approvals_data = run(client().get("/approvals"))
        print_jsonish(
            {
                "health": health,
                "session_conversation_id": conversation_id,
                "sessions": sessions_data.get("sessions", [])[:3],
                "pending_approvals": approvals_data.get("approvals", []),
            }
        )
        return True
    if command == "/deep":
        if not rest.strip():
            console.print("Usage: /deep <message>")
            return True
        _announce_slow_mode("deep")
        data = run(
            client().post(
                "/chat",
                {
                    "message": rest.strip(),
                    "conversation_id": conversation_id,
                    "mode": "deep",
                },
            )
        )
        _print_chat_result(data["result"])
        return True
    if command == "/council":
        if not rest.strip():
            console.print("Usage: /council <message>")
            return True
        _announce_slow_mode("council")
        data = run(
            client().post(
                "/chat",
                {
                    "message": rest.strip(),
                    "conversation_id": conversation_id,
                    "mode": "council",
                },
            )
        )
        _print_chat_result(data["result"])
        return True
    if command == "/approve":
        approval_id = rest.strip()
        if not approval_id:
            console.print("Usage: /approve <id>")
            return True
        print_jsonish(run(client().post("/tools/approve", {"approval_id": approval_id})))
        return True
    if command == "/deny":
        approval_id = rest.strip()
        if not approval_id:
            console.print("Usage: /deny <id>")
            return True
        print_jsonish(run(client().post("/tools/deny", {"approval_id": approval_id})))
        return True
    console.print(f"Unknown command: {command}")
    return True


def _close_session(session_id: str | None) -> None:
    """Best-effort session close so Archive reflection runs on REPL exit."""
    if not session_id:
        return
    try:
        run(client().post(f"/sessions/{session_id}/close", {}))
    except Exception:
        console.print("[yellow]Could not close the session; it will close on idle.[/yellow]")


def attach() -> None:
    """Bare `april`: join (or start) the current session and talk."""
    _maybe_autostart_daemon()
    data = run(client().post("/sessions", {"source": "terminal"}))
    conversation_id = data.get("conversation_id")
    session_id = data.get("session_id")
    joined = "existing" if data.get("joined_existing") else "new"
    console.print(f"Attached to {joined} APRIL session. Type /quit to exit.")
    try:
        while True:
            try:
                message = Prompt.ask("you")
            except EOFError:
                # Ctrl-D ends the REPL exactly like /quit.
                return
            if message.strip() in {"/quit", "/exit"}:
                return
            if not message.strip():
                continue
            if _handle_repl_command(message, conversation_id):
                continue
            response = run(
                client().post("/chat", {"message": message, "conversation_id": conversation_id})
            )
            _print_chat_result(response["result"])
    finally:
        # Every exit path (quit, Ctrl-D, Ctrl-C, errors) closes the session so
        # ArchiveReflectionService.reflect_session runs server-side.
        _close_session(session_id)


def one_shot(message: str) -> None:
    """`april <message>`: one terminal wake carrying a single command."""
    _maybe_autostart_daemon()
    data = run(client().post("/wake", {"source": "terminal", "text": message}))
    result = data.get("result")
    if result is not None:
        _print_chat_result(result)
    else:
        print_jsonish(data)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    listen: bool = typer.Option(
        False,
        "--listen",
        help="Hand this terminal to hands-free voice (same as `april voice listen`).",
    ),
) -> None:
    if ctx.invoked_subcommand is None:
        if listen:
            # Compatibility alias with terminal session continuity: this blocks
            # the terminal while voice wakes join the attached session.
            _terminal_voice_listen()
        else:
            attach()


def known_command_names() -> set[str]:
    """Names Typer can dispatch: used to tell subcommands from one-shot text."""
    import typer.main as typer_main

    command = typer_main.get_command(app)
    return set(getattr(command, "commands", {}).keys())


def main() -> None:
    """Entry point: bare attach, known subcommands, or one-shot message."""
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in known_command_names():
        message = " ".join(argv).strip()
        if message:
            one_shot(message)
            return
    app()


def _announce_slow_mode(mode: str) -> None:
    """Tell the user a slower rung was requested before waiting on it.

    Honest wording only: no exact timing is claimed because none is measured.
    """
    if mode == "deep":
        console.print(
            "[dim]Deep mode: I'll think about this more carefully. "
            "This local reasoning pass can take a while.[/dim]"
        )
    elif mode == "council":
        console.print(
            "[dim]Council mode: several local agents will answer and the best "
            "answer is selected. This can take a while.[/dim]"
        )


@app.command()
def ask(
    message: str,
    project_id: str | None = typer.Option(None, "--project-id"),
    repo_path: str | None = typer.Option(None, "--repo-path"),
    conversation_id: str | None = typer.Option(None, "--conversation-id"),
    mode: str = typer.Option("standard", "--mode", help="standard, deep, or council."),
) -> None:
    if mode not in _CHAT_MODES:
        raise typer.BadParameter("mode must be standard, deep, or council")
    _announce_slow_mode(mode)
    payload = {
        "message": message,
        "project_id": project_id,
        "repo_path": repo_path,
        "conversation_id": conversation_id,
        "mode": mode,
    }
    data = run(client().post("/chat", payload))
    result = data["result"]
    console.print(result["final_message"])
    if result.get("pending_approval"):
        console.print("[yellow]Approval required:[/yellow]")
        print_jsonish(result["pending_approval"])


@app.command()
def chat(
    project_id: str | None = typer.Option(None, "--project-id"),
    repo_path: str | None = typer.Option(None, "--repo-path"),
    mode: str = typer.Option("standard", "--mode", help="standard, deep, or council."),
) -> None:
    if mode not in _CHAT_MODES:
        raise typer.BadParameter("mode must be standard, deep, or council")
    console.print("APRIL chat. Type /quit to exit.")
    conversation_id = str(uuid.uuid4())
    while True:
        message = Prompt.ask("you")
        if message.strip() in {"/quit", "/exit"}:
            return
        if _handle_repl_command(message, conversation_id):
            continue
        ask(
            message,
            project_id=project_id,
            repo_path=repo_path,
            conversation_id=conversation_id,
            mode=mode,
        )


@app.command()
def models() -> None:
    data = run(client().get("/runtime/models"))
    print_models(data)


@model_app.command("load")
def model_load(model_id: str) -> None:
    data = run(client().post("/runtime/models/load", {"model_id": model_id}))
    print_jsonish(data)


@model_app.command("unload")
def model_unload(model_id: str) -> None:
    data = run(client().post("/runtime/models/unload", {"model_id": model_id}))
    print_jsonish(data)


@app.command()
def briefing() -> None:
    data = run(client().briefing())
    print_briefing(data)


@app.command()
def approvals() -> None:
    data = run(client().get("/approvals"))
    print_approvals(data)


@app.command()
def approve(approval_id: str) -> None:
    data = run(client().post("/tools/approve", {"approval_id": approval_id}))
    print_jsonish(data)


@app.command()
def deny(approval_id: str) -> None:
    data = run(client().post("/tools/deny", {"approval_id": approval_id}))
    print_jsonish(data)


@agent_app.command("pool")
def agent_pool() -> None:
    """Show the named agent pool: call signs plus persisted run/feedback stats."""
    print_jsonish(run(client().get("/pool/agents")))


@agent_app.command("run")
def agent_run(
    agent: str,
    message: str,
    project_id: str | None = typer.Option(None, "--project-id"),
    repo_path: str | None = typer.Option(None, "--repo-path"),
    conversation_id: str | None = typer.Option(None, "--conversation-id"),
) -> None:
    payload = {
        "agent": agent,
        "message": message,
        "project_id": project_id,
        "repo_path": repo_path,
        "conversation_id": conversation_id,
        "options": {"structured": True},
    }
    data = run(client().post("/agents/run", payload))
    result = data["result"]
    console.print(result["final_message"])
    if result.get("pending_approval"):
        console.print("[yellow]Approval required:[/yellow]")
        print_jsonish(result["pending_approval"])


@app.command()
def projects() -> None:
    data = run(client().get("/projects"))
    print_jsonish(data)


@project_app.command("add")
def project_add(path: str, name: str | None = None) -> None:
    data = run(client().post("/projects", {"path": path, "name": name}))
    print_jsonish(data)


@project_app.command("index")
def project_index(project_id: str) -> None:
    data = run(client().post(f"/projects/{project_id}/index", {}))
    print_jsonish(data)


@doc_app.command("add")
def doc_add(path: str) -> None:
    data = run(client().post("/documents", {"path": path}))
    print_jsonish(data)


@doc_app.command("list")
def doc_list() -> None:
    data = run(client().get("/documents"))
    print_jsonish(data)


@doc_app.command("search")
def doc_search(query: str) -> None:
    data = run(client().get("/documents/search", params={"q": query}))
    print_jsonish(data)


@memory_app.command("list")
def memory_list() -> None:
    data = run(client().get("/memory/search", params={"q": "*"}))
    print_jsonish(data)


@memory_app.command("search")
def memory_search(query: str) -> None:
    data = run(client().get("/memory/search", params={"q": query}))
    print_jsonish(data)


@memory_app.command("delete")
def memory_delete(memory_id: str) -> None:
    data = run(client().delete(f"/memory/{memory_id}"))
    print_jsonish(data)


@memory_app.command("export")
def memory_export() -> None:
    data = run(client().get("/memory/export"))
    print_jsonish(data)


@memory_app.command("inspect")
def memory_inspect(
    state: str = typer.Option(
        "machine",
        "--state",
        help="machine, superseded, expired, fading, or active.",
    ),
    limit: int = typer.Option(100, "--limit", min=1, max=500),
) -> None:
    """Inspect machine-written, superseded, expired, or fading memories."""
    data = run(client().get("/memory/inspect", params={"state": state, "limit": limit}))
    print_jsonish(data)


@memory_app.command("reindex")
def memory_reindex(
    wait: bool = typer.Option(False, "--wait"),
    json_output: bool = typer.Option(False, "--json"),
    wait_timeout: float = typer.Option(3600.0, "--wait-timeout", min=1.0, max=86400.0),
) -> None:
    job = run(client().post("/memory/reindex", {}))
    if wait:
        job = _wait_for_job(str(job["id"]), timeout_seconds=wait_timeout)
    if json_output:
        console.print_json(data=job)
        return
    console.print(f"Durable memory reindex job {job['id']} is {job['status']}.")
    if job.get("result"):
        print_jsonish(job["result"])
    else:
        console.print(f"  run april jobs show {job['id']}")
        console.print(f"  run april jobs cancel {job['id']}")


@memory_app.command("repair-index")
def memory_repair_index(
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Atomically repoint CURRENT and perform safe cleanup.",
    ),
) -> None:
    """Inspect recoverable generations; mutation is opt-in with ``--apply``."""
    data = run(client().post(f"/memory/repair-index?apply={str(apply).lower()}", {}))
    print_jsonish(data)


@conversation_app.command("delete")
def conversation_delete(conversation_id: str) -> None:
    data = run(client().delete(f"/conversations/{conversation_id}"))
    print_jsonish(data)


@reminder_app.command("list")
def reminder_list() -> None:
    data = run(client().get("/reminders"))
    print_jsonish(data)


@reminder_app.command("create")
def reminder_create(
    content: str,
    due_at: str | None = typer.Option(None, "--due-at"),
) -> None:
    data = run(client().post("/reminders", {"content": content, "due_at": due_at}))
    print_jsonish(data)


@reminder_app.command("delete")
def reminder_delete(reminder_id: str) -> None:
    data = run(client().delete(f"/reminders/{reminder_id}"))
    print_jsonish(data)


@task_app.command("list")
def task_list() -> None:
    data = run(client().get("/tasks"))
    print_jsonish(data)


# Import command modules after the Typer groups and compatibility helpers exist.
from apps.cli.commands.daemon import (
    daemon_install,
    daemon_start,
    daemon_status,
    daemon_stop,
    daemon_uninstall,
)
from apps.cli.commands.evolution import (
    evolve_adapter_activate,
    evolve_adapter_list,
    evolve_adapter_rollback,
    evolve_approve,
    evolve_dataset_export,
    evolve_diff,
    evolve_evals_pending,
    evolve_evals_promote,
    evolve_evals_reject,
    evolve_evals_show,
    evolve_history,
    evolve_off,
    evolve_on,
    evolve_pending,
    evolve_report,
    evolve_rollback,
    evolve_status,
    evolve_versions,
    playbook_adopt,
    playbook_list,
    playbook_mine,
    playbook_run,
)
from apps.cli.commands.jobs import (
    _wait_for_job,
    jobs_cancel,
    jobs_list,
    jobs_retry,
    jobs_show,
    jobs_submit,
)
from apps.cli.commands.system import bad, good, health, mute, sessions
from apps.cli.commands.voice import (
    _run_voice_listen,
    _terminal_voice_listen,
    voice_devices,
    voice_doctor_command,
    voice_enroll,
    voice_health_command,
    voice_listen,
    voice_ptt,
    voice_test_record,
    voice_test_stt,
    voice_test_tts,
)

if __name__ == "__main__":
    main()
