from __future__ import annotations

import anyio
import typer

from apps.cli.render import console
from april_common.audit import audit_logger_for_settings
from april_common.errors import AprilError, ConfigError
from april_common.settings import AprilSettings, load_settings
from services.memory.database import Database
from services.permissions.approvals import ApprovalStore

_RECOVERY_TOOL = "audit_recovery"


async def _validate_recovery_approval(
    settings: AprilSettings, approval_id: str, args: dict[str, object]
) -> None:
    database = Database(settings.database_path)
    await database.connect()
    try:
        approvals = ApprovalStore(
            database,
            audit_logger_for_settings(settings),
            expiry_seconds=settings.permissions.approval_expiry_seconds,
        )
        await approvals.validate_exact(
            approval_id=approval_id,
            tool=_RECOVERY_TOOL,
            args=args,
        )
    finally:
        await database.close()


async def _consume_recovery_approval(
    settings: AprilSettings,
    approval_id: str,
    args: dict[str, object],
    result: dict[str, object],
) -> None:
    database = Database(settings.database_path)
    await database.connect()
    try:
        audit = audit_logger_for_settings(settings)
        approvals = ApprovalStore(
            database,
            audit,
            expiry_seconds=settings.permissions.approval_expiry_seconds,
        )
        await approvals.consume_exact(
            approval_id=approval_id,
            tool=_RECOVERY_TOOL,
            args=args,
            result=result,
            actor="local-user",
            request_id=f"audit-recovery-{approval_id}",
        )
    finally:
        await database.close()


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
            raise RuntimeError(
                "production audit recovery requires an exact approved approval ID; "
                "this command never creates or bypasses approvals"
            )
        recovery_args = {"apply": True, "reason": reason.strip()}
        if apply and settings.environment == "production":
            assert approval_id is not None
            anyio.run(_validate_recovery_approval, settings, approval_id, recovery_args)
        result = audit_logger_for_settings(settings).recover(
            reason=reason,
            apply=apply,
            approval_id=approval_id,
        )
        if (
            apply
            and settings.environment == "production"
            and approval_id is not None
            and result.status == "recovered"
        ):
            anyio.run(
                _consume_recovery_approval,
                settings,
                approval_id,
                recovery_args,
                {"ok": True, "status": result.status},
            )
    except (AprilError, ConfigError, RuntimeError, ValueError) as exc:
        if json_output:
            reason_code = exc.code if isinstance(exc, AprilError) else type(exc).__name__
            console.print_json(data={"status": "refused", "reason_code": reason_code})
        else:
            if isinstance(exc, AprilError):
                console.print(f"[red]Audit recovery refused: {exc.code}.[/red]")
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
    if result.status == "unavailable":
        raise typer.Exit(1)
