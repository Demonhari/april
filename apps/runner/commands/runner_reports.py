from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.reports import (
    CleanResult,
    ReportListing,
    ReportSummary,
    clean_reports,
    known_report_types,
    latest_report,
    latest_report_of_type,
    list_report_summaries,
    summarize_path,
)
from apps.runner.wake_live import run_sentinel_live_verification

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


def _reports_dir() -> Path:
    return _composition_api._manager().home / "data" / "verification"


def _print_report_summary(summary: ReportSummary, *, with_actions: bool = True) -> None:
    console.print(
        f"[bold]{summary.basename}[/bold] — {summary.report_type} "
        f"(status={summary.status or 'n/a'})"
    )
    details = []
    if summary.generated_at:
        details.append(f"generated_at={summary.generated_at}")
    if summary.acceptance_level:
        details.append(f"level={summary.acceptance_level}")
    if summary.runtime_backend:
        details.append(f"backend={summary.runtime_backend}")
    if summary.services:
        details.append(f"services[{summary.services}]")
    if details:
        console.print("  " + ", ".join(details))
    if with_actions and summary.next_actions:
        console.print("  Next actions:")
        for action in summary.next_actions:
            console.print(f"    {action}", markup=False)


def _print_report_listing(listing: ReportListing) -> None:
    if not listing.reports:
        console.print(f"No reports found under {listing.directory}.")
        return
    table = Table(title=f"Verification reports ({listing.directory}, newest first)")
    table.add_column("Report")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Level")
    table.add_column("Generated at")
    for summary in listing.reports:
        table.add_row(
            summary.basename,
            summary.report_type,
            summary.status or "-",
            summary.acceptance_level or "-",
            summary.generated_at or "-",
        )
    console.print(table)


@_registry.reports_app.command("list")
def reports_list(json_output: bool = typer.Option(False, "--json")) -> None:
    """List redacted verification reports under data/verification, newest first."""
    listing = list_report_summaries(_reports_dir())
    if json_output:
        console.print_json(data=listing.model_dump())
    else:
        _print_report_listing(listing)


@_registry.reports_app.command("latest")
def reports_latest(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show the newest report of any known type."""
    summary = latest_report(_reports_dir())
    if summary is None:
        console.print("No known verification reports found under data/verification.")
        raise typer.Exit(1)
    if json_output:
        console.print_json(data=summary.model_dump())
    else:
        _print_report_summary(summary)


@_registry.reports_app.command("show")
def reports_show(
    path: Path,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show a concise, redacted summary of a single report JSON file."""
    summary = summarize_path(path)
    if summary is None:
        console.print(f"[red]Could not read a JSON report at {path.name}.[/red]")
        raise typer.Exit(1)
    if json_output:
        console.print_json(data=summary.model_dump())
    else:
        _print_report_summary(summary)


@_registry.reports_app.command("show-latest")
def reports_show_latest(
    report_type: str = typer.Option(..., "--type", help="Report type to show the latest of."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show the newest report of a specific type (e.g. acceptance, mac_activation)."""
    known = tuple(known_report_types())
    if report_type not in known:
        console.print(f"[red]Unknown report type '{report_type}'. Known: {', '.join(known)}.[/red]")
        raise typer.Exit(1)
    summary = latest_report_of_type(_reports_dir(), report_type)
    if summary is None:
        console.print(f"No {report_type} report found under data/verification.")
        raise typer.Exit(1)
    if json_output:
        console.print_json(data=summary.model_dump())
    else:
        _print_report_summary(summary)


@_registry.reports_app.command("clean")
def reports_clean(
    older_than_days: int = typer.Option(..., "--older-than-days", min=0),
    dry_run: bool = typer.Option(False, "--dry-run"),
    apply_changes: bool = typer.Option(False, "--apply"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Delete report JSON files older than N days (dry-run by default).

    Only ``*.json`` files directly inside data/verification are ever touched, and
    deletion happens only with --apply. Nothing outside data/verification is removed.
    """
    if apply_changes and dry_run:
        console.print("[red]Use either --apply or --dry-run, not both.[/red]")
        raise typer.Exit(1)
    result = clean_reports(_reports_dir(), older_than_days=older_than_days, apply=apply_changes)
    if json_output:
        console.print_json(data=result.model_dump())
    else:
        _print_reports_clean(result)


def _print_reports_clean(result: CleanResult) -> None:
    label = "Deleted" if result.applied else "Would delete"
    console.print(
        f"{label} {len(result.candidates)} report(s) older than {result.older_than_days} day(s) "
        f"in {result.directory}."
    )
    for candidate in result.candidates:
        console.print(f"  {candidate.basename} (age {candidate.age_days}d)")
    if not result.applied and result.candidates:
        console.print("Re-run with --apply to delete these report files.")
