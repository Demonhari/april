from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import typer

from apps.cli.groups import voice_app
from apps.cli.render import console, print_jsonish
from april_common.settings import get_settings


def client() -> Any:
    from apps.cli import main as cli_main

    return cli_main.client()


def run(coro: Any) -> Any:
    from apps.cli import main as cli_main

    return cli_main.run(coro)


def _maybe_autostart_daemon() -> None:
    from apps.cli import main as cli_main

    cli_main._maybe_autostart_daemon()


def _close_session(session_id: str | None) -> None:
    from apps.cli import main as cli_main

    cli_main._close_session(session_id)


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

    Samples are stored only on this Mac for the configured local ONNX speaker
    verifier. Enrollment never changes ``wake.speaker_gate`` by itself, and
    soft mode is only a convenience filter, never authentication.
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
            "note": (
                "enrollment does not enable speaker_gate; configure "
                "wake.speaker_verifier_model_path before using soft mode"
            ),
        }
    )


@voice_app.command("listen")
def voice_listen() -> None:
    _terminal_voice_listen()


def _run_voice_listen(*, session_hint: str | None = None) -> None:
    if session_hint is None:
        raise typer.BadParameter("resident voice attachment requires a session")
    from services.wake.control import attach_resident_sentinel

    try:
        with attach_resident_sentinel(get_settings(), session_hint=session_hint) as attachment:
            print_jsonish(attachment.status)
            console.print("Attached to resident Sentinel. Press Enter or Ctrl-D to stop.")
            with contextlib.suppress(EOFError):
                input()
    except (OSError, RuntimeError) as exc:
        console.print(
            "[red]Resident Sentinel is unavailable or degraded: "
            f"{exc}. Check `run april daemon status` and `run april voice health`.[/red]"
        )
        raise typer.Exit(1) from exc


def _terminal_voice_listen() -> None:
    _maybe_autostart_daemon()
    data = run(client().post("/sessions", {"source": "terminal"}))
    session_id = data.get("session_id")
    try:
        _run_voice_listen(session_hint=session_id if isinstance(session_id, str) else None)
    finally:
        _close_session(session_id if isinstance(session_id, str) else None)
