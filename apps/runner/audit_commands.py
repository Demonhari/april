from __future__ import annotations

import typer

from apps.cli.render import console
from april_common.audit import audit_logger_for_settings
from april_common.errors import AprilError, ConfigError
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
    if result.status == "unavailable":
        # Unavailable is distinct from a corrupt chain, but it is still a
        # failed verification for shell callers and readiness gates.
        raise typer.Exit(2)


@audit_app.command("recover")
def recover_audit(
    apply: bool = typer.Option(False, "--apply", help="Apply recovery after reviewing the plan."),
    approve: bool = typer.Option(
        False, "--approve", help="Record owner consent for a reviewed plan."
    ),
    reason: str = typer.Option(
        "owner-approved local recovery",
        "--reason",
        help="Bounded reason recorded in the new chain.",
    ),
    approval_id: str | None = typer.Option(None, "--approval-id"),
    plan_id: str | None = typer.Option(None, "--plan-id"),
    plan_digest: str | None = typer.Option(None, "--plan-digest"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Plan, consent to, apply, or resume a corrupt local audit-chain recovery."""
    try:
        settings = load_settings()
        if apply and approve:
            raise RuntimeError("Use either --approve or --apply, not both.")
        audit = audit_logger_for_settings(settings)
        if approve:
            if not plan_id:
                raise RuntimeError("--approve requires --plan-id from a prior recovery plan.")
            approval_result = audit.approve_recovery(plan_id=plan_id, plan_digest=plan_digest)
            if json_output:
                console.print_json(data=approval_result)
            else:
                console.print(f"Audit recovery consent recorded for plan {plan_id}.")
                console.print(f"Approval ID: {approval_result['approval_id']}")
            return
        if apply and settings.environment == "production" and (not approval_id or not plan_id):
            raise RuntimeError(
                "production audit recovery requires --plan-id and an approved recovery "
                "approval ID; this command never creates or bypasses approvals"
            )
        result = audit.recover(
            reason=reason,
            apply=apply,
            approval_id=approval_id,
            plan_id=plan_id,
        )
    except (AprilError, ConfigError, RuntimeError, ValueError) as exc:
        if json_output:
            reason_code = exc.code if isinstance(exc, AprilError) else type(exc).__name__
            details = dict(exc.details) if isinstance(exc, AprilError) else {}
            status = (
                "incomplete"
                if (
                    isinstance(exc, AprilError)
                    and (
                        details.get("log_changed") is True
                        or exc.code == "AUDIT_RECOVERY_INCOMPLETE"
                        or details.get("phase")
                        in {
                            "log_publication",
                            "journal_log_publication",
                            "anchor_publication",
                            "journal_anchor_publication",
                            "verification",
                            "journal_finalization",
                        }
                    )
                )
                else "refused"
            )
            payload = {"status": status, "reason_code": reason_code}
            for key in (
                "phase",
                "log_changed",
                "anchor_state",
                "plan_id",
                "approval_id",
                "resume_command",
            ):
                if key in details:
                    payload[key] = details[key]
            console.print_json(data=payload)
        else:
            if isinstance(exc, AprilError):
                label = (
                    "incomplete"
                    if (
                        exc.details.get("log_changed")
                        or exc.code == "AUDIT_RECOVERY_INCOMPLETE"
                        or exc.details.get("phase")
                        in {
                            "log_publication",
                            "journal_log_publication",
                            "anchor_publication",
                            "journal_anchor_publication",
                            "verification",
                            "journal_finalization",
                        }
                    )
                    else "refused"
                )
                console.print(f"[red]Audit recovery {label}: {exc.code}.[/red]")
                if exc.details.get("resume_command"):
                    console.print(f"Resume with: {exc.details['resume_command']}")
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
