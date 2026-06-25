from __future__ import annotations

import asyncio
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import typer
from rich.prompt import Prompt

from apps.cli.client import ApiOfflineError, AprilApiClient
from apps.cli.render import (
    console,
    print_approvals,
    print_briefing,
    print_jsonish,
    print_models,
)
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
doc_app = typer.Typer(help="Document operations.")
app.add_typer(model_app, name="model")
app.add_typer(project_app, name="project")
app.add_typer(memory_app, name="memory")
app.add_typer(voice_app, name="voice")
app.add_typer(conversation_app, name="conversation")
app.add_typer(agent_app, name="agent")
app.add_typer(reminder_app, name="reminder")
app.add_typer(task_app, name="task")
app.add_typer(doc_app, name="doc")


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


@memory_app.command("reindex")
def memory_reindex() -> None:
    console.print("Reindexing vector memory under the current embedding provider...")
    data = run(client().post("/memory/reindex", {}))
    console.print(
        f"Reindexed {data['reindexed']} records using "
        f"{data['provider']} ({data['dimensions']} dimensions)."
    )


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
    from april_common.errors import RuntimeUnavailableError
    from services.voice.conversation_loop import PushToTalkLoop, interactive_capture_strategy
    from services.voice.health import voice_health
    from services.voice.microphone import SoundDeviceMicrophone

    settings = get_settings()
    health_report = voice_health(settings)
    if health_report.status == "degraded":
        console.print(health_report.model_dump())

    if seconds is not None:
        # Deterministic fixed-duration mode for scripts and smoke tests.
        console.print(f"Recording for {seconds:.1f}s. Speak now.")
        loop = PushToTalkLoop(api_client=client(), record_seconds=seconds)
    else:
        # Interactive, stop-controlled push-to-talk (Enter to start, Enter to stop).
        microphone = SoundDeviceMicrophone(
            device=settings.voice.input_device,
            max_seconds=settings.voice.max_record_seconds,
        )
        capture = interactive_capture_strategy(
            microphone,
            max_seconds=settings.voice.max_record_seconds,
            read_line=input,
            announce=console.print,
        )
        loop = PushToTalkLoop(api_client=client(), microphone=microphone, capture=capture)

    try:
        answer = run(loop.run_once())
    except KeyboardInterrupt:
        console.print("Push-to-talk cancelled; microphone released.")
        raise typer.Exit(130) from None
    except (ValueError, RuntimeUnavailableError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(answer)


@voice_app.command("health")
def voice_health_command() -> None:
    from services.voice.health import voice_health

    print_jsonish(voice_health(get_settings()).model_dump())


@voice_app.command("doctor")
def voice_doctor_command() -> None:
    from services.voice.health import voice_doctor

    report = voice_doctor(get_settings())
    print_jsonish(report)
    if report["status"] != "ok":
        console.print("Voice listen will fall back to push-to-talk until missing components exist.")


@voice_app.command("devices")
def voice_devices() -> None:
    from services.voice.health import query_audio_devices

    print_jsonish(query_audio_devices())


@voice_app.command("test-record")
def voice_test_record(seconds: float = typer.Option(3.0, "--seconds", min=0.1, max=30.0)) -> None:
    from services.voice.microphone import SoundDeviceMicrophone

    settings = get_settings()
    output_path = settings.audio_cache_path / "test-record.wav"
    mic = SoundDeviceMicrophone(
        device=settings.voice.input_device,
        max_seconds=seconds,
    )
    try:
        recorded = run(mic.record_push_to_talk(output_path))
    finally:
        if not settings.voice.retain_debug_audio:
            output_path.unlink(missing_ok=True)
    print_jsonish(
        {
            "recorded": True,
            "seconds": seconds,
            "path": str(recorded),
            "retained": settings.voice.retain_debug_audio,
        }
    )


@voice_app.command("test-stt")
def voice_test_stt(audio_path: Path) -> None:
    from services.voice.speech_to_text import WhisperCppSpeechToText

    settings = get_settings()
    stt = WhisperCppSpeechToText(
        settings.voice.whisper_binary_path,
        settings.voice.whisper_model_path,
    )
    text = run(stt.transcribe(audio_path.expanduser().resolve()))
    print_jsonish({"text": text})


@voice_app.command("test-tts")
def voice_test_tts(text: str) -> None:
    from services.voice.text_to_speech import PiperTextToSpeech

    settings = get_settings()
    output_path = settings.audio_cache_path / "test-tts.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts = PiperTextToSpeech(settings.voice.piper_binary_path, settings.voice.piper_model_path)
    synthesized = run(tts.synthesize(text, output_path))
    retained = settings.voice.retain_debug_audio
    if not retained:
        synthesized.unlink(missing_ok=True)
    print_jsonish({"synthesized": True, "path": str(synthesized), "retained": retained})


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
