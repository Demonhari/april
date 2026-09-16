"""CLI commands for reviewed, local third-party source integration."""

from __future__ import annotations

from pathlib import Path

import typer

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.third_party_sources import (
    ThirdPartySourceError,
    build_colibri,
    inspect_source_manifest,
    run_colibri_command,
)
from april_common.settings import load_settings


@_registry.third_party_app.command("doctor")
def third_party_doctor(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect staged third-party provenance without network access."""

    settings = load_settings()
    result = inspect_source_manifest(settings.home)
    if json_output:
        console.print_json(data=result)
    else:
        status = "ready" if result.get("ready") else "not ready"
        console.print(f"Third-party source doctor: {status}")
        reason = result.get("reason")
        if isinstance(reason, str):
            console.print(f"Reason: {reason}")
        for entry in result.get("entries", []):
            if isinstance(entry, dict):
                console.print(f"- {entry.get('id')}: {entry.get('status')} ({entry.get('reason')})")
    if not result.get("ready"):
        raise typer.Exit(1)


@_registry.third_party_app.command("build-colibri")
def build_colibri_command(
    engine: str = typer.Option("qwen36", "--engine"),
    arch: str = typer.Option("native", "--arch"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Build only already-staged Colibri source with its local Makefile."""

    settings = load_settings()
    try:
        result = build_colibri(settings.home, engine=engine, arch=arch)
    except ThirdPartySourceError as exc:
        console.print(f"[red]Colibri build unavailable: {exc}[/red]")
        raise typer.Exit(1) from exc
    if json_output:
        console.print_json(data=result)
    else:
        console.print("Built staged Colibri source; APRIL did not start the server.")


def _run_colibri(
    subcommand: str,
    model: Path,
    options: list[str],
) -> None:
    settings = load_settings()
    try:
        return_code = run_colibri_command(
            settings.home,
            subcommand,
            model=model,
            options=options,
        )
    except ThirdPartySourceError as exc:
        console.print(f"[red]Colibri {subcommand} unavailable: {exc}[/red]")
        raise typer.Exit(1) from exc
    if return_code:
        raise typer.Exit(return_code)


@_registry.third_party_app.command("colibri-info")
def colibri_info(
    model: Path = typer.Option(..., "--model", exists=False),
) -> None:
    """Run Colibri's local model-info diagnostic."""

    _run_colibri("info", model, [])


@_registry.third_party_app.command("colibri-doctor")
def colibri_doctor(
    model: Path = typer.Option(..., "--model", exists=False),
    gpu: str = typer.Option("none", "--gpu"),
    ram: int | None = typer.Option(None, "--ram"),
    ctx: int | None = typer.Option(None, "--ctx"),
    deep: bool = typer.Option(False, "--deep"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run Colibri's local doctor without starting the model server."""

    options = ["--gpu", gpu]
    if ram is not None:
        options.extend(("--ram", str(ram)))
    if ctx is not None:
        options.extend(("--ctx", str(ctx)))
    if deep:
        options.append("--deep")
    if json_output:
        options.append("--json")
    _run_colibri("doctor", model, options)


@_registry.third_party_app.command("colibri-plan")
def colibri_plan(
    model: Path = typer.Option(..., "--model", exists=False),
    ram: int = typer.Option(..., "--ram"),
    ctx: int = typer.Option(..., "--ctx"),
    gpu: str = typer.Option("none", "--gpu"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Print Colibri's local resource plan."""

    options = ["--ram", str(ram), "--ctx", str(ctx), "--gpu", gpu]
    if json_output:
        options.append("--json")
    _run_colibri("plan", model, options)


@_registry.third_party_app.command("colibri-serve")
def colibri_serve(
    model: Path = typer.Option(..., "--model", exists=False),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
    model_id: str | None = typer.Option(None, "--model-id"),
) -> None:
    """Run Colibri's explicitly requested loopback server."""

    options = ["--host", "127.0.0.1", "--port", str(port)]
    if model_id:
        options.extend(("--model-id", model_id))
    _run_colibri("serve", model, options)
