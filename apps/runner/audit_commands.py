from __future__ import annotations

import typer

from apps.cli.render import console
from april_common.audit import audit_logger_for_settings
from april_common.errors import ConfigError
from april_common.settings import load_settings

audit_app = typer.Typer(help="Verify APRIL's local hash-chained audit log.")


@audit_app.command("verify")
def verify_audit(json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        settings = load_settings()
        result = audit_logger_for_settings(settings).verify()
    except (ConfigError, RuntimeError) as exc:
        if json_output:
            console.print_json(
                data={
                    "status": "unavailable",
                    "valid": False,
                    "corrupt": False,
                    "error": type(exc).__name__,
                }
            )
        else:
            console.print("[red]Audit verification is unavailable.[/red]")
        raise typer.Exit(2) from exc
    if json_output:
        console.print_json(data=result.to_dict())
    else:
        console.print(
            f"Audit chain: {result.status}; records={result.record_count}; "
            f"terminal_sequence={result.terminal_sequence or 0}"
        )
        for issue in result.issues:
            location = f" line {issue.line}" if issue.line is not None else ""
            console.print(f"  {issue.code}{location}: {issue.detail}")
    if result.corrupt:
        raise typer.Exit(1)


@audit_app.command("recover")
def recover_audit(
    apply: bool = typer.Option(False, "--apply", help="Apply recovery after reviewing the plan."),
    reason: str = typer.Option(
        "owner-approved local recovery",
        "--reason",
        help="Bounded reason recorded in the new chain.",
    ),
    approval_id: str | None = typer.Option(None, "--approval-id"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Dry-run or explicitly recover a corrupt local audit chain."""
    try:
        settings = load_settings()
        if apply and settings.environment == "production" and not approval_id:
            raise RuntimeError("production audit recovery requires an exact approval ID")
        result = audit_logger_for_settings(settings).recover(reason=reason, apply=apply)
    except (ConfigError, RuntimeError, ValueError) as exc:
        if json_output:
            console.print_json(data={"status": "refused", "reason_code": type(exc).__name__})
        else:
            console.print("[red]Audit recovery refused.[/red]")
        raise typer.Exit(1) from exc
    if json_output:
        console.print_json(data=result.to_dict())
    else:
        issues = ",".join(result.issue_codes) or "none"
        console.print(f"Audit recovery: {result.status}; issues={issues}.")
        if result.status == "dry_run":
            console.print("No files changed. Review, then add --apply explicitly.")
    if result.status == "dry_run" and apply:
        raise typer.Exit(1)
