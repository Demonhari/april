from __future__ import annotations

import asyncio
import uuid
from collections.abc import Coroutine
from typing import Any

import typer
from rich.prompt import Prompt

from apps.cli.client import ApiOfflineError, AprilApiClient
from apps.cli.render import console, print_approvals, print_jsonish, print_models
from april_common.settings import get_settings

app = typer.Typer(help="APRIL local assistant CLI.")
model_app = typer.Typer(help="Model operations.")
project_app = typer.Typer(help="Project operations.")
memory_app = typer.Typer(help="Memory operations.")
voice_app = typer.Typer(help="Voice operations.")
conversation_app = typer.Typer(help="Conversation operations.")
agent_app = typer.Typer(help="Direct specialist agent operations.")
reminder_app = typer.Typer(help="Reminder operations.")
task_app = typer.Typer(help="Task inspection operations.")
app.add_typer(model_app, name="model")
app.add_typer(project_app, name="project")
app.add_typer(memory_app, name="memory")
app.add_typer(voice_app, name="voice")
app.add_typer(conversation_app, name="conversation")
app.add_typer(agent_app, name="agent")
app.add_typer(reminder_app, name="reminder")
app.add_typer(task_app, name="task")


def client() -> AprilApiClient:
    settings = get_settings()
    return AprilApiClient(
        f"http://{settings.api.host}:{settings.api.port}",
        settings.api.token,
        timeout=settings.runtime.request_timeout_seconds,
    )


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        return asyncio.run(coro)
    except ApiOfflineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def health() -> None:
    data = run(client().get("/health", auth=False))
    print_jsonish(data)


@app.command()
def ask(
    message: str,
    project_id: str | None = typer.Option(None, "--project-id"),
    repo_path: str | None = typer.Option(None, "--repo-path"),
    conversation_id: str | None = typer.Option(None, "--conversation-id"),
) -> None:
    payload = {
        "message": message,
        "project_id": project_id,
        "repo_path": repo_path,
        "conversation_id": conversation_id,
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
) -> None:
    console.print("APRIL chat. Type /quit to exit.")
    conversation_id = str(uuid.uuid4())
    while True:
        message = Prompt.ask("you")
        if message.strip() in {"/quit", "/exit"}:
            return
        ask(
            message,
            project_id=project_id,
            repo_path=repo_path,
            conversation_id=conversation_id,
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


@voice_app.command("ptt")
def voice_ptt(seconds: float | None = typer.Option(None, "--seconds", min=0.1, max=300.0)) -> None:
    from services.voice.conversation_loop import PushToTalkLoop
    from services.voice.health import voice_health

    settings = get_settings()
    health_report = voice_health(settings)
    if health_report.status == "degraded":
        console.print(health_report.model_dump())
    console.print("Recording. Speak after this prompt.")
    loop = PushToTalkLoop(api_client=client(), record_seconds=seconds)
    run(loop.run_once())


@voice_app.command("health")
def voice_health_command() -> None:
    from services.voice.health import voice_health

    print_jsonish(voice_health(get_settings()).model_dump())


@voice_app.command("doctor")
def voice_doctor_command() -> None:
    from services.voice.health import voice_health

    report = voice_health(get_settings()).model_dump()
    print_jsonish(report)
    if report["status"] != "ok":
        console.print("Voice listen will fall back to push-to-talk until missing components exist.")


@voice_app.command("devices")
def voice_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        console.print("[red]sounddevice is not installed. Install APRIL voice extras.[/red]")
        raise typer.Exit(1) from exc
    print_jsonish({"devices": str(sd.query_devices())})


@voice_app.command("listen")
def voice_listen() -> None:
    from services.voice.conversation_loop import WakeWordConversationLoop
    from services.voice.health import voice_health

    settings = get_settings()
    health_report = voice_health(settings)
    if health_report.status == "degraded":
        console.print(health_report.model_dump())
    loop = WakeWordConversationLoop(api_client=client())
    run(loop.run_forever())


if __name__ == "__main__":
    app()
