from __future__ import annotations

import asyncio
import sys
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
daemon_app = typer.Typer(help="Daemon operations.")
playbook_app = typer.Typer(help="Playbook operations.")
evolve_app = typer.Typer(help="Evolution operations.")
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

_DAEMON_AUTOSTART_REPORTED = False
_CHAT_MODES = {"standard", "deep", "council"}


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


def _maybe_autostart_daemon() -> None:
    """Best-effort apriald autostart before attach/one-shot, when configured."""
    global _DAEMON_AUTOSTART_REPORTED
    settings = get_settings()
    if not settings.daemon.autostart_on_cli:
        return
    try:
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
        autostart_if_needed(settings)
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
            # Compatibility alias: `april --listen` starts the Sentinel loop.
            # No shell hooks are installed; this simply blocks this terminal.
            voice_listen()
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


@app.command()
def health() -> None:
    data = run(client().get("/health", auth=False))
    print_jsonish(data)


@app.command()
def mute(
    off: bool = typer.Option(False, "--off", help="Release the hard mute."),
) -> None:
    """Hard-mute the Sentinel: the microphone stream is fully released."""
    from services.wake.sentinel import MuteSwitch

    switch = MuteSwitch(get_settings().mute_flag_path)
    if off:
        switch.unmute()
        console.print("Voice unmuted. The Sentinel may reopen the microphone.")
    else:
        switch.mute()
        console.print("Voice hard-muted. The Sentinel releases the microphone.")


@app.command()
def sessions() -> None:
    """List recent wake/conversation sessions."""
    data = run(client().get("/sessions"))
    print_jsonish(data)


@app.command()
def good() -> None:
    """Mark the last answer in the active session as good."""
    data = run(client().post("/feedback", {"rating": "good"}))
    print_jsonish(data)


@app.command()
def bad(
    reason: str = typer.Argument(None, help="Optional short reason."),
) -> None:
    """Mark the last answer in the active session as bad."""
    data = run(client().post("/feedback", {"rating": "bad", "reason": reason}))
    print_jsonish(data)


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


@voice_app.command("enroll")
def voice_enroll(
    samples: int = typer.Option(3, "--samples", min=1, max=10),
    seconds: float = typer.Option(3.0, "--seconds", min=1.0, max=15.0),
) -> None:
    """Record local speaker enrollment samples under data/voice_profiles/.

    Samples are stored only on this Mac for a future speaker gate. The gate
    itself stays `wake.speaker_gate: off` until a local verifier model exists;
    enrollment never changes wake behaviour by itself.
    """
    from april_common.errors import RuntimeUnavailableError
    from services.voice.microphone import SoundDeviceMicrophone

    settings = get_settings()
    profile_dir = settings.resolve_path(Path("data/voice_profiles"))
    profile_dir.mkdir(parents=True, exist_ok=True)
    microphone = SoundDeviceMicrophone(
        device=settings.voice.input_device,
        max_seconds=seconds,
    )
    recorded: list[str] = []
    for index in range(1, samples + 1):
        console.print(f"Sample {index}/{samples}: say the wake phrase. Recording {seconds:.0f}s...")
        output_path = profile_dir / f"enroll-{index:02d}.wav"
        try:
            run(microphone.record_push_to_talk(output_path))
        except (RuntimeUnavailableError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        recorded.append(output_path.name)
    print_jsonish(
        {
            "enrolled_samples": recorded,
            "profile_dir": str(profile_dir),
            "speaker_gate": settings.wake.speaker_gate,
            "note": "speaker_gate stays off until a local verifier is configured",
        }
    )


@voice_app.command("listen")
def voice_listen() -> None:
    from april_common.errors import RuntimeUnavailableError
    from services.voice.health import voice_health
    from services.wake.sentinel import run_sentinel

    settings = get_settings()
    health_report = voice_health(settings)
    if health_report.status == "degraded":
        console.print(health_report.model_dump())
    try:
        run(run_sentinel(settings))
    except RuntimeUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@daemon_app.command("install")
def daemon_install() -> None:
    from apps.daemon.launchd import LaunchdManager

    path = LaunchdManager(get_settings()).install()
    print_jsonish({"installed": True, "plist_path": str(path)})


@daemon_app.command("uninstall")
def daemon_uninstall() -> None:
    from apps.daemon.launchd import LaunchdManager

    removed = LaunchdManager(get_settings()).uninstall()
    print_jsonish({"removed": removed})


@daemon_app.command("start")
def daemon_start() -> None:
    from apps.daemon.apriald import start_daemon_background

    print_jsonish(start_daemon_background(get_settings()))


@daemon_app.command("stop")
def daemon_stop() -> None:
    from apps.daemon.apriald import stop_daemon

    print_jsonish(stop_daemon(get_settings()))


@daemon_app.command("status")
def daemon_status() -> None:
    from apps.daemon.apriald import read_daemon_status
    from apps.daemon.launchd import LaunchdManager

    settings = get_settings()
    status = read_daemon_status(settings)
    status["launchd"] = LaunchdManager(settings).status()
    print_jsonish(status)


@playbook_app.command("list")
def playbook_list() -> None:
    print_jsonish(run(client().get("/playbooks")))


@playbook_app.command("run")
def playbook_run(
    playbook_id: str,
    project_id: str | None = typer.Option(None, "--project-id"),
    conversation_id: str | None = typer.Option(None, "--conversation-id"),
) -> None:
    payload = {"project_id": project_id, "conversation_id": conversation_id}
    print_jsonish(run(client().post(f"/playbooks/{playbook_id}/run", payload)))


@playbook_app.command("adopt")
def playbook_adopt(path: Path) -> None:
    import json

    import yaml

    from skills.playbooks import PlaybookDefinition

    resolved = path.expanduser().resolve()
    if resolved.suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    playbook = PlaybookDefinition.model_validate(payload)
    print_jsonish(run(client().post("/playbooks/adopt", playbook.model_dump())))


@evolve_app.command("versions")
def evolve_versions(agent: str | None = typer.Option(None, "--agent")) -> None:
    params = {"agent": agent} if agent else None
    print_jsonish(run(client().get("/evolution/versions", params=params)))


@evolve_app.command("rollback")
def evolve_rollback(agent: str, version: int) -> None:
    print_jsonish(run(client().post("/evolution/rollback", {"agent": agent, "version": version})))


@evolve_app.command("report")
def evolve_report() -> None:
    print_jsonish(run(client().get("/evolution/report/latest")))


@evolve_app.command("status")
def evolve_status() -> None:
    """Show evolution enablement, kill switch, last run, and overlay counts."""
    print_jsonish(run(client().get("/evolution/status")))


@evolve_app.command("history")
def evolve_history(limit: int = typer.Option(20, "--limit", min=1, max=200)) -> None:
    """List past Dreamer runs, newest first."""
    print_jsonish(run(client().get("/evolution/history", params={"limit": limit})))


@evolve_app.command("diff")
def evolve_diff(
    agent: str,
    from_version: int | None = typer.Option(None, "--from", min=1),
    to_version: int | None = typer.Option(None, "--to", min=1),
) -> None:
    """Unified diff between two prompt-overlay versions of one agent."""
    params: dict[str, Any] = {"agent": agent}
    if from_version is not None:
        params["from_version"] = from_version
    if to_version is not None:
        params["to_version"] = to_version
    data = run(client().get("/evolution/diff", params=params))
    if data.get("diff"):
        console.print(data["diff"])
    else:
        print_jsonish(data)


@evolve_app.command("off")
def evolve_off() -> None:
    """Set the local kill switch: the Dreamer never runs while it is active."""
    data = run(client().post("/evolution/off", {}))
    print_jsonish(data)
    console.print("Evolution is now hard-disabled. Re-enable with: april evolve on")


@evolve_app.command("on")
def evolve_on() -> None:
    """Clear the local kill switch (config evolution.enabled still applies)."""
    print_jsonish(run(client().post("/evolution/on", {})))


@evolve_app.command("pending")
def evolve_pending() -> None:
    """List write-capable agent overlays awaiting explicit approval."""
    print_jsonish(run(client().get("/evolution/overlays/pending")))


@evolve_app.command("approve")
def evolve_approve(agent: str, content_hash: str) -> None:
    """Approve one pending overlay by agent and exact SHA-256 content hash."""
    data = run(
        client().post(
            "/evolution/overlays/approve",
            {"agent": agent, "content_hash": content_hash},
        )
    )
    print_jsonish(data)


evolve_dataset_app = typer.Typer(help="Fine-tuning dataset operations (export only).")
evolve_app.add_typer(evolve_dataset_app, name="dataset")


@evolve_dataset_app.command("export")
def evolve_dataset_export(
    name: str | None = typer.Option(None, "--name", help="Dataset basename."),
) -> None:
    """Export the reviewable JSONL fine-tune dataset under data/evolution/datasets."""
    data = run(client().post("/evolution/dataset/export", {"name": name}))
    print_jsonish(data)


if __name__ == "__main__":
    main()
