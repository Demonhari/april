from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from apps.cli.render import console
from apps.runner.speaker_live import (
    disable_speaker_gate,
    enable_soft_speaker_gate,
    run_speaker_live_verification,
)
from april_common.settings import load_settings

speaker_gate_app = typer.Typer(help="Optional local speaker convenience gate.")


def register_speaker_commands(voice_app: typer.Typer) -> None:
    @voice_app.command("verify-speaker-live")
    def verify_speaker_live(
        report: Path = typer.Option(
            Path("data/verification/speaker-live.json"),
            "--report",
        ),
        debug_capture: bool = typer.Option(False, "--debug-capture"),
    ) -> None:
        settings = load_settings()
        target = report if report.is_absolute() else settings.home / report
        result = asyncio.run(
            run_speaker_live_verification(
                settings=settings,
                confirm_capture=typer.confirm,
                retain_debug_audio=debug_capture,
                report_path=target,
            )
        )
        console.print_json(data=result.model_dump(mode="json"))
        if not result.speaker_live_verified:
            raise typer.Exit(1)

    voice_app.add_typer(speaker_gate_app, name="speaker-gate")


@speaker_gate_app.command("enable-soft")
def enable_soft(
    report: Path = typer.Option(
        Path("data/verification/speaker-live.json"),
        "--report",
    ),
) -> None:
    settings = load_settings()
    target = report if report.is_absolute() else settings.home / report
    enable_soft_speaker_gate(settings, target)
    console.print("Speaker gate set to soft using a fresh successful local report.")


@speaker_gate_app.command("disable")
def disable() -> None:
    settings = load_settings()
    disable_speaker_gate(settings)
    console.print("Speaker gate disabled.")
