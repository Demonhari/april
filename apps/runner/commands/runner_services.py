from __future__ import annotations

from typing import TypeVar

import typer

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.wake_live import run_sentinel_live_verification

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


@_registry.april_app.command()
def start(
    ctx: typer.Context,
    preflight: bool = typer.Option(
        False, "--preflight", help="Run startup preflight and refuse to start unless it passes."
    ),
    fake: bool = typer.Option(False, "--fake", help="Allow/start with the fake runtime backend."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Start APRIL's services, optionally gated by a safe startup preflight.

    With ``--preflight`` the services are started only when every preflight check
    passes; preflight never starts a service, loads a model, or opens the mic.
    """
    use_fake = _composition_api._effective_fake(ctx, fake)
    if preflight:
        report = _composition_api.build_preflight_report(
            _composition_api._manager().home, fake=use_fake
        )
        if json_output:
            console.print_json(data=report.model_dump())
        else:
            _composition_api._print_preflight(report)
        if not report.ok:
            if not json_output:
                console.print(
                    "[red]Preflight failed; services were not started.[/red] "
                    f"Blockers: {', '.join(report.failures)}"
                )
            raise typer.Exit(1)
    _composition_api._print_status(_composition_api._ensure_services(use_fake))


@_registry.april_app.command()
def stop() -> None:
    _composition_api._print_status(_composition_api._manager().stop())


@_registry.april_app.command()
def restart(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start services with fake runtime."),
) -> None:
    _composition_api._print_status(
        _composition_api._manager().restart(
            fake_backend=_composition_api._effective_fake(ctx, fake)
        )
    )


@_registry.april_app.command()
def logs(
    lines: int = typer.Option(80, "--lines", min=1, max=1000),
    tail: int | None = typer.Option(None, "--tail", min=1, max=1000),
) -> None:
    console.print(_composition_api._manager().logs(lines=tail or lines))
