from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.daily_driver import DailyDriverReport, build_daily_driver_report
from apps.runner.multi_model_report import (
    write_multi_model_report,
)
from apps.runner.readiness import ReadinessReport
from apps.runner.verify import (
    build_workflow_report,
    write_workflow_report,
)
from apps.runner.wake_live import run_sentinel_live_verification
from april_common.config_fingerprint import config_fingerprint_digest

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


_DAILY_STATUS_STYLE = {
    "ready": "[green]ready[/green]",
    "warning": "[yellow]warning[/yellow]",
    "blocker": "[red]blocker[/red]",
    "not_run": "[dim]not_run[/dim]",
}

_READINESS_STATUS_STYLE = {
    "ok": "[green]ok[/green]",
    "warning": "[yellow]warning[/yellow]",
    "blocker": "[red]blocker[/red]",
    "skipped": "[dim]skipped[/dim]",
}


@_registry.april_app.callback(invoke_without_command=True)
def april(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    oneshot: bool = typer.Option(
        False,
        "--oneshot",
        help="Stop services after the delegated command when this invocation started them.",
    ),
) -> None:
    ctx.obj = {"fake": fake, "oneshot": oneshot}
    if ctx.invoked_subcommand is None:
        _composition_api._delegate(["chat"], fake=fake, oneshot=oneshot)


@_registry.april_app.command()
def chat(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["chat"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.april_app.command()
def ask(
    ctx: typer.Context,
    message: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    project_id: str | None = typer.Option(None, "--project-id"),
    repo_path: str | None = typer.Option(None, "--repo-path"),
) -> None:
    args = ["ask", message]
    if project_id:
        args.extend(["--project-id", project_id])
    if repo_path:
        args.extend(["--repo-path", repo_path])
    _composition_api._delegate(
        args,
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.april_app.command()
def status(json_output: bool = typer.Option(False, "--json")) -> None:
    status_value = _composition_api._manager().status()
    if json_output:
        console.print_json(data=_composition_api._status_payload(status_value))
        return
    _composition_api._print_status(status_value)


@_registry.app.command()
def doctor(
    daily_driver: bool = typer.Option(
        False, "--daily-driver", help="Summarize daily-driver readiness instead of the launcher."
    ),
    json_output: bool = typer.Option(False, "--json"),
    run_real_checks: bool = typer.Option(
        False,
        "--run-real-checks",
        help="Run real model + workflow verification first (loads models; opt-in).",
    ),
) -> None:
    if daily_driver:
        _daily_driver(json_output=json_output, run_real_checks=run_real_checks)
        return
    _composition_api._doctor()


@_registry.april_app.command("doctor")
def april_doctor(
    daily_driver: bool = typer.Option(
        False, "--daily-driver", help="Summarize daily-driver readiness instead of the launcher."
    ),
    json_output: bool = typer.Option(False, "--json"),
    run_real_checks: bool = typer.Option(
        False,
        "--run-real-checks",
        help="Run real model + workflow verification first (loads models; opt-in).",
    ),
) -> None:
    if daily_driver:
        _daily_driver(json_output=json_output, run_real_checks=run_real_checks)
        return
    _composition_api._doctor()


def _daily_driver(*, json_output: bool, run_real_checks: bool) -> None:
    home = _composition_api._manager().home
    if run_real_checks:
        _run_real_daily_checks(home)
    report = build_daily_driver_report(home)
    if json_output:
        console.print_json(data=report.model_dump())
    else:
        _print_daily_driver(report)
    if report.overall == "blocker":
        raise typer.Exit(1)


def _run_real_daily_checks(home: Path) -> None:
    """Opt-in heavy path: run real-model + workflow verification, writing reports.

    Skips cleanly (with a printed note) when the local prerequisites — a non-fake
    backend with llama-cpp-python and configured GGUFs present — are not met, so
    `--run-real-checks` never blocks on a machine that cannot support real models.
    """
    readiness = _composition_api.build_readiness_report(home)
    chat_models = [
        model
        for model in readiness.models
        if model.backend == "llama_cpp" and model.role != "embedding"
    ]
    blocked = (
        readiness.runtime_is_fake
        or not readiness.llama_cpp_python_available
        or not chat_models
        or any(not model.path_exists for model in chat_models)
    )
    if blocked:
        console.print(
            "[yellow]--run-real-checks skipped: real model prerequisites are not met "
            "(backend/llama-cpp/GGUFs). Summarizing existing reports only.[/yellow]"
        )
        return
    reports_dir = home / "data" / "verification"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint_digest(home)
    console.print("[bold]Running real model verification (this loads models)…[/bold]")
    previous = os.environ.get("APRIL_VERIFY_ROUTING_EVALS")
    os.environ["APRIL_VERIFY_ROUTING_EVALS"] = "1"
    try:
        verifier = _composition_api.run_all_configured_models_verification(
            home, require_real_model=True
        )
        write_multi_model_report(
            verifier.build_report(config_fingerprint=fingerprint),
            reports_dir / "mac-readiness.json",
        )
    finally:
        if previous is None:
            os.environ.pop("APRIL_VERIFY_ROUTING_EVALS", None)
        else:
            os.environ["APRIL_VERIFY_ROUTING_EVALS"] = previous
    console.print("[bold]Running real workflow verification…[/bold]")
    checks = _composition_api.run_workflow_verification(home, real_model=True)
    write_workflow_report(
        build_workflow_report(checks, real_model_requested=True, config_fingerprint=fingerprint),
        reports_dir / "workflow-real.json",
    )


def _print_daily_driver(report: DailyDriverReport) -> None:
    overall = _DAILY_STATUS_STYLE.get(report.overall, report.overall)
    console.print(f"APRIL daily-driver doctor — {overall}")
    console.print("")
    core = _DAILY_STATUS_STYLE.get(report.core_real_model, report.core_real_model)
    console.print(f"Core real model: {core}")
    console.print(
        "Workflow security: "
        f"{_DAILY_STATUS_STYLE.get(report.workflow_security, report.workflow_security)}"
    )
    hardened = _DAILY_STATUS_STYLE.get(report.hardened_go_live, report.hardened_go_live)
    console.print(f"Hardened go-live: {hardened}")
    if report.hardened_reason:
        console.print(f"Reason: {report.hardened_reason}", markup=False)
    table = Table(title="Daily-driver checks")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row(
            check.name,
            _DAILY_STATUS_STYLE.get(check.status, check.status),
            check.detail,
        )
    console.print(table)
    if report.next_commands:
        console.print("[bold]Next commands:[/bold]")
        for command in report.next_commands:
            console.print(f"  {command}", markup=False)


@_registry.april_app.command("readiness")
def readiness(json_output: bool = typer.Option(False, "--json")) -> None:
    """Explain offline exactly what is missing for real local-model readiness.

    Reads only configs/env; never starts a service, loads a model, opens the
    microphone, downloads anything, or installs anything. Prints actionable
    commands only. Paths and tokens are redacted.
    """
    report = _composition_api.build_readiness_report(_composition_api._manager().home)
    if json_output:
        console.print_json(data=report.model_dump())
        return
    _print_readiness(report)


def _print_readiness(report: ReadinessReport) -> None:
    headline = (
        "[green]preflight ready[/green]"
        if report.real_model_preflight_ready
        else "[red]preflight blocked[/red]"
    )
    console.print(
        "APRIL readiness — "
        f"{headline}; real verification not run "
        f"(backend={report.runtime_backend}, env={report.environment})"
    )
    table = Table(title="Readiness checks")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row(
            check.name,
            _READINESS_STATUS_STYLE.get(check.status, check.status),
            check.detail,
        )
    console.print(table)
    if report.next_actions:
        console.print("[bold]Next actions (run these yourself; nothing is run for you):[/bold]")
        for action in report.next_actions:
            # markup=False so tokens like '.[runtime]' are not parsed as Rich tags.
            console.print(f"  {action}", markup=False)
    if not report.blockers:
        console.print(
            "[green]No preflight blockers: run the real verification command to confirm.[/green]"
        )


@_registry.april_app.command()
def health(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["health"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.april_app.command()
def models(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["models"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.april_app.command()
def briefing(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["briefing"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.april_app.command()
def desktop(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    native: bool = typer.Option(
        False,
        "--native",
        help="Open a native window via the optional [desktop] extra (pywebview) "
        "instead of the default browser.",
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Resolve services and the local URL but do not open anything.",
    ),
) -> None:
    """Launch the local Desktop UI over authenticated loopback HTTP.

    Never starts voice, wake-word, or the microphone. The API token is passed via
    the URL fragment (browser) or the JS bridge (native), never as a query string.
    """
    _composition_api._ensure_services(_composition_api._effective_fake(ctx, fake))
    manager = _composition_api._manager()
    token = manager.settings.api.token
    base_url = _composition_api._desktop_base_url(manager)
    if not token:
        console.print("[red]No API token configured. Run `run april setup tokens` first.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]APRIL Desktop is available at {base_url}[/green]")
    console.print("The API token is passed locally (URL fragment / JS bridge) and never logged.")
    if no_open:
        return
    if native:
        if _composition_api._open_desktop_native(base_url, token):
            return
        console.print(
            "[yellow]pywebview is not installed. Install the optional native window with "
            "`pip install -e '.[desktop]'`, or use the default browser path.[/yellow]"
        )
    fragment_url = f"{base_url}#token={token}"
    if not _composition_api._open_desktop_browser(fragment_url):
        console.print(
            f"[yellow]Could not open a browser automatically. Open {base_url} and append "
            "your token as #token=... in the address bar.[/yellow]"
        )


@_registry.april_app.command()
def approvals(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["approvals"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.april_app.command()
def approve(
    ctx: typer.Context,
    approval_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["approve", approval_id],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.april_app.command()
def deny(
    ctx: typer.Context,
    approval_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["deny", approval_id],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.april_app.command()
def projects(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["projects"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )
