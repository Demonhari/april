"""CLI commands for reviewed, local third-party source integration."""

from __future__ import annotations

import typer

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.third_party_sources import (
    ThirdPartySourceError,
    build_colibri,
    inspect_source_manifest,
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
    jobs: int = typer.Option(1, "--jobs", min=1, max=64),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Build only already-staged Colibri source with the local CMake toolchain."""

    settings = load_settings()
    try:
        result = build_colibri(settings.home, jobs=jobs)
    except ThirdPartySourceError as exc:
        console.print(f"[red]Colibri build unavailable: {exc}[/red]")
        raise typer.Exit(1) from exc
    if json_output:
        console.print_json(data=result)
    else:
        console.print("Built staged Colibri source; APRIL did not start the server.")
