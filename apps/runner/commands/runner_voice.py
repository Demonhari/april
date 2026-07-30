from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypeVar

import typer

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.wake_live import run_sentinel_live_verification

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


@_registry.voice_app.command("health")
def voice_health(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["voice", "health"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.voice_app.command("doctor")
def voice_doctor(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["voice", "doctor"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.voice_app.command("verify-live")
def voice_verify_live(
    report: Path | None = typer.Option(
        None, "--report", help="Write a redacted live voice verification report JSON here."
    ),
    seconds: float = typer.Option(3.0, "--seconds", min=0.2, max=10.0),
    retain_debug_audio: bool = typer.Option(
        False,
        "--retain-debug-audio",
        help="Keep the exact temporary audio files created by this explicit verification run.",
    ),
) -> None:
    settings = _composition_api._manager().settings
    doctor = _composition_api.collect_voice_doctor(settings)
    console.print(f"Voice doctor status: {doctor['status']}")
    guidance = doctor.get("macos_microphone_permission_guidance")
    if guidance:
        console.print(str(guidance))
    console.print("Wake-word listening is not used by this verification.")

    def confirm(message: str) -> bool:
        return typer.confirm(message, default=False)

    def show_transcript(transcript: str) -> None:
        console.print("Local whisper.cpp transcript:")
        console.print(transcript if transcript else "[yellow]<empty>[/yellow]")

    result = asyncio.run(
        _composition_api.run_voice_live_verification(
            settings=settings,
            confirm_recording=confirm,
            confirm_transcription=confirm,
            confirm_playback=confirm,
            seconds=seconds,
            retain_debug_audio=retain_debug_audio,
            transcript_observer=show_transcript,
            report_path=report,
        )
    )
    console.print(
        "Voice live verification: "
        f"{result.summary} (recording={result.recording_success}, "
        f"stt={result.stt_success}, transcript_length={result.transcript_length}, "
        f"tts={result.tts_success}, playback_confirmed={result.playback_user_confirmed})"
    )
    if report is not None:
        console.print(f"[green]Wrote voice verification report to {report.expanduser()}[/green]")
    if result.summary != "pass":
        raise typer.Exit(1)


@_registry.voice_app.command("verify-wake-live")
def voice_verify_wake_live(
    report: Path | None = typer.Option(
        None, "--report", help="Write a redacted live wake-word verification report JSON here."
    ),
    wake_wait_seconds: float | None = typer.Option(
        None, "--wake-wait-seconds", min=1.0, max=120.0, help="How long to wait for the wake word."
    ),
    utterance_max_seconds: float | None = typer.Option(
        None, "--utterance-max-seconds", min=1.0, max=60.0, help="Max command length after wake."
    ),
    retain_debug_audio: bool = typer.Option(
        False,
        "--retain-debug-audio",
        help="Keep the exact temporary audio files created by this explicit verification run.",
    ),
) -> None:
    """Verify the live Sentinel wake-word ('April') path on this Mac.

    Start APRIL services first (``run april`` or ``run april --fake``) so the
    Core ``/wake`` endpoint can be reached during verification. This command
    requires microphone access, whisper.cpp, a local wake-word model, and the
    loopback API; missing artifacts are reported as blockers, not passes.
    """
    settings = _composition_api._manager().settings
    doctor = _composition_api.collect_voice_doctor(settings)
    console.print(f"Voice doctor status: {doctor['status']}")
    for key in ("macos_microphone_permission_guidance", "wake_word_guidance"):
        guidance = doctor.get(key)
        if guidance:
            console.print(str(guidance))
    if settings.voice.wake_word_model_path is None:
        console.print(
            "[yellow]No wake-word model is configured.[/yellow] Configure one with "
            "`run april setup voice --wake-word-model /absolute/path/april.onnx` first."
        )

    def confirm(message: str) -> bool:
        return typer.confirm(message, default=False)

    result = asyncio.run(
        _composition_api.run_sentinel_live_verification(
            settings=settings,
            confirm_microphone=confirm,
            wake_wait_seconds=wake_wait_seconds,
            report_path=report,
        )
    )
    console.print(
        "Sentinel live verification: "
        f"{result.summary} (wake_word_detected={result.wake_word_detected}, "
        f"recording={result.recording_success}, stt={result.stt_success}, "
        f"transcript_length={result.transcript_length}, "
        f"normalized_transcript_length={result.normalized_transcript_length}, "
        f"api={result.api_success})"
    )
    for skipped in result.skipped:
        console.print(f"[yellow]Skipped {skipped.name}:[/yellow] {skipped.reason}")
    if report is not None:
        console.print(
            f"[green]Wrote wake-word verification report to {report.expanduser()}[/green]"
        )
    if result.summary != "pass":
        raise typer.Exit(1)


@_registry.voice_app.command("verify-conversation-live")
def voice_verify_conversation_live(
    report: Path = typer.Option(
        Path("data/verification/voice-conversation-live.json"),
        "--report",
        help="Write the redacted two-turn live conversation report here.",
    ),
    timeout_seconds: float = typer.Option(
        180.0,
        "--timeout-seconds",
        min=30.0,
        max=600.0,
        help="Overall operator interaction timeout.",
    ),
    retain_debug_audio: bool = typer.Option(
        False,
        "--retain-debug-audio",
        help="Keep temporary utterance/reply audio for this explicit run.",
    ),
) -> None:
    """Verify two endpointed turns and real production barge-in on this Mac."""
    settings = _composition_api._manager().settings
    console.print(
        "Turn 1: say “April” and a request, including a natural 300-500 ms pause. "
        "When playback begins, use the configured barge-in trigger and complete Turn 2."
    )
    console.print(
        "The report stores timing, lengths, booleans, and reason codes only—never "
        "transcripts, responses, audio, tokens, device names, or absolute paths."
    )

    result = asyncio.run(
        _composition_api.run_voice_conversation_live_verification(
            settings=settings,
            confirm_microphone=lambda message: typer.confirm(message, default=False),
            report_path=report,
            timeout_seconds=timeout_seconds,
            retain_debug_audio=retain_debug_audio,
        )
    )
    console.print(
        "Voice conversation live verification: "
        f"{result.summary} (turns={result.turn_count}, "
        f"same_conversation={result.same_conversation}, "
        f"barge_in={result.barge_in_detected}, "
        f"verified={result.voice_conversation_live_verified})"
    )
    console.print(f"[green]Wrote voice conversation report to {report.expanduser()}[/green]")
    if not result.voice_conversation_live_verified:
        raise typer.Exit(1)


@_registry.voice_app.command("devices")
def voice_devices(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["voice", "devices"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.voice_app.command("ptt")
def voice_ptt(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    seconds: float | None = typer.Option(None, "--seconds", min=0.1, max=300.0),
) -> None:
    args = ["voice", "ptt"]
    if seconds is not None:
        args.extend(["--seconds", str(seconds)])
    _composition_api._delegate(
        args,
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.voice_app.command("test-record")
def voice_test_record(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    seconds: float = typer.Option(3.0, "--seconds", min=0.1, max=30.0),
) -> None:
    _composition_api._delegate(
        ["voice", "test-record", "--seconds", str(seconds)],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.voice_app.command("test-stt")
def voice_test_stt(
    ctx: typer.Context,
    audio_path: Path,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["voice", "test-stt", str(audio_path)],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.voice_app.command("test-tts")
def voice_test_tts(
    ctx: typer.Context,
    text: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["voice", "test-tts", text],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.voice_app.command("listen")
def voice_listen(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["voice", "listen"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )
