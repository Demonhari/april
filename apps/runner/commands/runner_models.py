from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.model_downloads import (
    ModelDownloadReport,
    default_model_download_report_path,
    write_model_download_report,
)
from apps.runner.model_tools import (
    apply_model_profile,
    load_model_profiles,
    model_doctor,
    recommend_model_profile,
)
from apps.runner.wake_live import run_sentinel_live_verification
from april_common.errors import ConfigError

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


@_registry.model_app.command("load")
def model_load(
    ctx: typer.Context,
    model_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["model", "load", model_id],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.model_app.command("unload")
def model_unload(
    ctx: typer.Context,
    model_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["model", "unload", model_id],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.model_app.command("doctor")
def model_doctor_command(json_output: bool = typer.Option(False, "--json")) -> None:
    payload = model_doctor(_composition_api._manager().home)
    if json_output:
        console.print_json(data=payload)
        return
    _composition_api._print_model_doctor(payload)


@_registry.model_app.command("recommend")
def model_recommend_command(json_output: bool = typer.Option(False, "--json")) -> None:
    """Report a non-mutating model-profile recommendation for this Mac.

    Inspects only local hardware. It never installs, downloads, switches
    configuration, edits shell files, or sends data anywhere.
    """
    payload = recommend_model_profile(_composition_api._manager().home)
    if json_output:
        console.print_json(data=payload)
        return
    _composition_api._print_model_recommendation(payload)


def _print_model_download(report: ModelDownloadReport) -> None:
    heading = "APPLIED" if report.applied else "DRY RUN"
    console.print(f"APRIL model download — {heading}")
    console.print(
        "Download only installs local GGUF files and updates model config; "
        "it does not verify real model readiness."
    )
    for entry in report.entries:
        suffix = f", sha256={entry.sha256}" if entry.sha256 else ""
        console.print(
            f"{entry.role}: {entry.repo_id}/{entry.filename} -> "
            f"{entry.target_path} ({entry.status}{suffix})"
        )
    if report.registration_applied:
        console.print(
            "Model registry updated"
            + (
                f" (backup={report.registration_backup_basename})"
                if report.registration_backup_basename
                else ""
            )
        )
    console.print("Real model verified: false")
    console.print("Next commands:")
    for command in report.next_commands:
        console.print(f"  {command}", markup=False)


@_registry.model_app.command("download", context_settings={"allow_extra_args": True})
def model_download_command(
    ctx: typer.Context,
    all_core: bool = typer.Option(False, "--all-core", help="Download brain/coding/reading."),
    role: str | None = typer.Option(
        None, "--role", help="Download one manifest role: brain/coding/reading/reasoning/embedding."
    ),
    apply_changes: bool = typer.Option(False, "--apply"),
    yes: bool = typer.Option(False, "--yes", help="Confirm non-interactive network download."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing target file."),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Skip roles whose target file already exists."
    ),
    write_report: bool = typer.Option(
        False,
        "--write-report",
        help="Write a redacted report to data/verification/model-download-<timestamp>.json.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect the legacy model manifest without downloading or registering models."""
    if apply_changes:
        console.print(
            "[red]Model download/apply is retired. Obtain the GGUF manually, calculate "
            "its SHA-256 independently, then use exact-approved "
            "`run april model import ... --sha256 EXPECTED_SHA256`.[/red]"
        )
        raise typer.Exit(1)
    confirmed = yes
    try:
        report = _composition_api.run_model_downloads(
            _composition_api._manager().home,
            all_core=all_core,
            role=role,
            apply=apply_changes,
            yes=confirmed,
            force=force,
            skip_existing=skip_existing,
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if ctx.args and not write_report:
        console.print("[red]Unexpected extra argument. Did you mean --write-report PATH?[/red]")
        raise typer.Exit(1)
    if len(ctx.args) > 1:
        console.print("[red]Only one --write-report path may be supplied.[/red]")
        raise typer.Exit(1)
    if json_output:
        console.print_json(data=report.model_dump())
    else:
        _print_model_download(report)
    if write_report:
        target = (
            Path(ctx.args[0])
            if ctx.args
            else default_model_download_report_path(_composition_api._manager().home)
        )
        written = write_model_download_report(report, target)
        console.print(f"[green]Wrote model download report to {written}[/green]")


@_registry.model_app.command("benchmark")
def model_benchmark_command(
    ctx: typer.Context,
    model_id: str,
    wait: bool = typer.Option(False, "--wait"),
    json_output: bool = typer.Option(False, "--json"),
    wait_timeout: float = typer.Option(3600.0, "--wait-timeout", min=1.0, max=86400.0),
) -> None:
    payload = json.dumps({"model_id": model_id}, separators=(",", ":"))
    args = ["jobs", "submit", "model_benchmark", "--payload", payload]
    if wait:
        args.append("--wait")
    if json_output:
        args.append("--json")
    if wait_timeout != 3600.0:
        args.extend(["--wait-timeout", str(wait_timeout)])
    _composition_api._delegate(
        args,
        fake=_composition_api._effective_fake(ctx, False),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.model_app.command("verify")
def model_verify_command(
    ctx: typer.Context,
    model_id: str,
    wait: bool = typer.Option(False, "--wait"),
    json_output: bool = typer.Option(False, "--json"),
    wait_timeout: float = typer.Option(900.0, "--wait-timeout", min=1.0, max=86400.0),
) -> None:
    """Submit explicit verification of an already registered local model."""
    payload = json.dumps({"model_id": model_id}, separators=(",", ":"))
    args = ["jobs", "submit", "model_import_verification", "--payload", payload]
    if wait:
        args.append("--wait")
    if json_output:
        args.append("--json")
    if wait_timeout != 900.0:
        args.extend(["--wait-timeout", str(wait_timeout)])
    _composition_api._delegate(
        args,
        fake=_composition_api._effective_fake(ctx, False),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.profile_app.command("list")
def model_profile_list() -> None:
    profiles = load_model_profiles(_composition_api._manager().home)
    table = Table(title="APRIL Model Profiles")
    table.add_column("Profile")
    table.add_column("Description")
    for name, profile in profiles.items():
        description = profile.get("description", "") if isinstance(profile, dict) else ""
        table.add_row(str(name), str(description))
    console.print(table)


@_registry.profile_app.command("apply")
def model_profile_apply(profile_name: str) -> None:
    try:
        backup = apply_model_profile(
            home=_composition_api._manager().home, profile_name=profile_name
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Applied model profile: {profile_name}[/green]")
    console.print(f"Backup: {backup}")
