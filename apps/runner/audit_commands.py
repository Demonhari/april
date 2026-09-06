from __future__ import annotations

import typer

from apps.cli.render import console
from april_common.audit import AuditLogger, audit_logger_for_settings
from april_common.errors import AprilError, ConfigError
from april_common.settings import AprilSettings, load_settings

audit_app = typer.Typer(help="Verify APRIL's local hash-chained audit log.")


def _approval_command(plan_id: str | None, plan_digest: str | None) -> str | None:
    if not plan_id or not plan_digest:
        return None
    prefix = "run april audit recover --approve"
    return f"{prefix} --plan-id {plan_id} --plan-digest {plan_digest} --json"


def _apply_command(plan_id: str | None, approval_id: str | None) -> str | None:
    if not plan_id or not approval_id:
        return None
    return f"run april audit recover --apply --plan-id {plan_id} --approval-id {approval_id} --json"


def _verification_commands() -> list[str]:
    return [
        "run april audit verify --json",
        "run april readiness",
        "run april start --preflight",
    ]


def _quarantine_location(settings: AprilSettings, audit: AuditLogger, directory: str | None) -> str:
    """Display the actual recovery root without exposing an unrelated absolute path."""
    if not directory:
        return "unavailable"
    recovery_root = audit.recovery_root
    location = (recovery_root / directory).resolve()
    home = settings.home.resolve()
    try:
        return str(location.relative_to(home))
    except ValueError:
        # Custom audit paths can place the recovery root outside APRIL_HOME.
        # Keep the operator-useful location, while JSON reports retain their
        # existing path-redacted shape.
        return str(location)


@audit_app.command("verify")
def verify_audit(json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        settings = load_settings()
        result = audit_logger_for_settings(settings).verify()
    except (AprilError, ConfigError, OSError, RuntimeError) as exc:
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
                payload = dict(approval_result)
                returned_approval_id = approval_result.get("approval_id")
                next_command = _apply_command(
                    plan_id,
                    returned_approval_id if isinstance(returned_approval_id, str) else None,
                )
                if next_command is not None:
                    payload["next_commands"] = [next_command]
                console.print_json(data=payload)
            else:
                console.print(f"Audit recovery consent recorded for plan {plan_id}.")
                console.print(f"Approval ID: {approval_result['approval_id']}")
                console.print(
                    "Apply command: "
                    f"run april audit recover --apply --plan-id {plan_id} "
                    f"--approval-id {approval_result['approval_id']} --json",
                    markup=False,
                )
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
            if status == "incomplete" and isinstance(payload.get("resume_command"), str):
                payload["next_commands"] = [payload["resume_command"]]
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
        payload = result.to_dict()
        next_commands: list[str] = []
        if result.status == "dry_run":
            approval = _approval_command(result.plan_id, result.plan_digest)
            if approval is not None:
                next_commands.append(approval)
        elif result.status == "recovered":
            next_commands.extend(_verification_commands())
        if next_commands:
            payload["next_commands"] = next_commands
        console.print_json(data=payload)
    else:
        issues = ",".join(result.issue_codes) or "none"
        console.print(f"Audit recovery: {result.status}; issues={issues}.")
        if result.status == "dry_run":
            console.print(
                "The active audit log and protected anchor were not changed. "
                "A quarantine backup, recovery plan, and recovery-journal entry were created."
            )
            console.print(
                f"Plan ID: {result.plan_id}\n"
                f"Plan digest: {result.plan_digest}\n"
                f"Expires at: {result.expires_at}\n"
                f"Record count: {result.record_count}\n"
                f"Issue codes: {issues}\n"
                f"Original log SHA-256: {result.quarantined_log_sha256}\n"
                f"Quarantine: {_quarantine_location(settings, audit, result.quarantine_directory)}"
            )
            console.print(
                "Applying creates a NEW audit chain; the original bytes remain preserved "
                "as unverified historical evidence."
            )
            console.print(
                f"Approval command: {_approval_command(result.plan_id, result.plan_digest)}",
                markup=False,
            )
        elif result.status == "recovered":
            console.print(
                "The new audit chain and protected anchor verified successfully. "
                "The quarantined original remains unverified historical evidence."
            )
            console.print("Next: run april audit verify --json", markup=False)
            console.print("Next: run april readiness", markup=False)
            console.print("Next: run april start --preflight", markup=False)
    if result.status == "dry_run" and apply:
        raise typer.Exit(1)
    if result.status == "unavailable":
        raise typer.Exit(1)
