from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.bootstrap import bootstrap as run_bootstrap
from apps.runner.evals import run_fake_brain_eval, run_real_brain_eval
from apps.runner.install import is_april_wrapper, path_contains_dir
from apps.runner.mac_report import ReportThresholds, write_report
from apps.runner.model_tools import (
    apply_model_profile,
    create_macos_app_stub,
    import_model,
    load_model_profiles,
    model_doctor,
    recommend_model_profile,
    setup_model_set,
    setup_voice_stack,
)
from apps.runner.multi_model_report import write_multi_model_report
from apps.runner.readiness import ReadinessReport, build_readiness_report
from apps.runner.service_manager import AprilServiceManager, ServiceStatus
from apps.runner.soak import run_fake_soak, write_soak_report
from apps.runner.verify import (
    BenchmarkResult,
    TargetMacValidator,
    VerifyCheck,
    build_workflow_report,
    run_all_configured_models_verification,
    run_fake_verification,
    run_model_benchmark,
    run_real_model_verification,
    run_workflow_verification,
    write_workflow_report,
)
from apps.runner.voice_live import run_voice_live_verification
from april_common.config_validation import validate_configuration
from april_common.effective_config import load_agents_file, load_permissions_file, load_tools_file
from april_common.errors import ConfigError
from april_common.settings import load_settings
from april_common.token_setup import generate_tokens, write_token_env_file
from services.april_runtime.client import RuntimeClient
from services.april_runtime.model_registry import ModelRegistry
from services.memory.database import Database
from services.memory.embeddings import HashedTokenEmbedding
from services.memory.migrations import run_migrations
from services.memory.user_profile import UserProfileStore
from services.voice.health import voice_doctor as collect_voice_doctor

_T = TypeVar("_T")

app = typer.Typer(help="Global command dispatcher.")
april_app = typer.Typer(help="Run APRIL from any folder.", invoke_without_command=True)
model_app = typer.Typer(help="Model operations.")
project_app = typer.Typer(help="Project operations.")
memory_app = typer.Typer(help="Memory operations.")
conversation_app = typer.Typer(help="Conversation operations.")
config_app = typer.Typer(help="Configuration operations.")
agent_app = typer.Typer(help="Direct specialist agent operations.")
voice_app = typer.Typer(help="Voice operations.")
reminder_app = typer.Typer(help="Reminder operations.")
task_app = typer.Typer(help="Task inspection operations.")
eval_app = typer.Typer(help="Local evaluation operations.")
setup_app = typer.Typer(help="Local setup utilities.")
user_profile_app = typer.Typer(help="Local user-profile operations.")
app.add_typer(april_app, name="april")
april_app.add_typer(model_app, name="model")
april_app.add_typer(project_app, name="project")
april_app.add_typer(memory_app, name="memory")
april_app.add_typer(conversation_app, name="conversation")
april_app.add_typer(config_app, name="config")
april_app.add_typer(agent_app, name="agent")
april_app.add_typer(voice_app, name="voice")
april_app.add_typer(reminder_app, name="reminder")
april_app.add_typer(task_app, name="task")
april_app.add_typer(eval_app, name="eval")
april_app.add_typer(setup_app, name="setup")
april_app.add_typer(user_profile_app, name="profile")


def _manager() -> AprilServiceManager:
    return AprilServiceManager()


def _desktop_base_url(manager: AprilServiceManager) -> str:
    settings = manager.settings
    return f"http://{settings.api.host}:{settings.api.port}/desktop"


def _open_desktop_browser(url: str) -> bool:
    # Token travels in the URL fragment only; fragments are never sent to the
    # server, and the SPA strips it from the address bar immediately on load.
    return webbrowser.open(url, new=2)


class DesktopTokenBridge:
    """Minimal pywebview JS API: the page may only fetch the API token.

    Exposed to the page as ``window.pywebview.api``. Keeping the surface to a
    single ``get_token`` method means the SPA cannot reach arbitrary Python.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def get_token(self) -> str:
        return self._token


def _open_desktop_native(url: str, token: str) -> bool:
    # Optional native window via the [desktop] extra (pywebview). The token is
    # delivered only through the async JS bridge (window.pywebview.api.get_token),
    # never via a URL, the page HTML, or an injected global. Returns False when
    # pywebview is not installed so the caller can fall back to the browser.
    try:
        import webview
    except ImportError:
        return False

    webview.create_window("APRIL Desktop", url, js_api=DesktopTokenBridge(token))
    webview.start()
    return True


def _effective_fake(ctx: typer.Context, explicit: bool) -> bool:
    inherited = bool((ctx.obj or {}).get("fake", False))
    return inherited or explicit


def _effective_oneshot(ctx: typer.Context) -> bool:
    return bool((ctx.obj or {}).get("oneshot", False))


def _run_april_cli(args: list[str]) -> int:
    env = dict(os.environ)
    env.setdefault("APRIL_HOME", str(_manager().home))
    completed = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", *args],
        cwd=env["APRIL_HOME"],
        env=env,
        check=False,
    )
    return completed.returncode


def _ensure_services(fake: bool) -> ServiceStatus:
    try:
        status = _manager().start(fake_backend=fake)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not status.ok:
        console.print("[red]APRIL services are not healthy.[/red]")
        _print_status(status)
        raise typer.Exit(1)
    return status


def _delegate(args: list[str], *, fake: bool, oneshot: bool = False) -> None:
    manager = _manager()
    before = manager.status()
    try:
        _ensure_services(fake)
        if oneshot:
            console.print(
                "[yellow]APRIL oneshot mode: services will stop after this command.[/yellow]"
            )
        else:
            console.print("[green]APRIL services are running and will remain running.[/green]")
        code = _run_april_cli(args)
    finally:
        if oneshot and not before.ok:
            console.print("[yellow]Stopping APRIL services started for oneshot mode.[/yellow]")
            _print_status(manager.stop())
    raise typer.Exit(code)


def _print_status(status: ServiceStatus) -> None:
    table = Table(title="APRIL Services")
    table.add_column("Service")
    table.add_column("PID")
    table.add_column("Running")
    table.add_column("Healthy")
    table.add_column("Log")
    for info in (status.runtime, status.api):
        table.add_row(
            info.name,
            str(info.pid or "-"),
            "yes" if info.running else "no",
            "yes" if info.healthy else "no",
            str(info.log_path),
        )
    console.print(table)


def _status_payload(status: ServiceStatus) -> dict[str, Any]:
    return {
        "runtime": {
            "pid": status.runtime.pid,
            "running": status.runtime.running,
            "healthy": status.runtime.healthy,
            "log_path": str(status.runtime.log_path),
        },
        "api": {
            "pid": status.api.pid,
            "running": status.api.running,
            "healthy": status.api.healthy,
            "log_path": str(status.api.log_path),
        },
        "ok": status.ok,
    }


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve() or left.samefile(right)
    except FileNotFoundError:
        return left.resolve() == right.resolve()


def _doctor() -> None:
    manager = _manager()
    home = manager.home
    local_bin = Path.home() / ".local" / "bin"
    run_path = local_bin / "run"
    april_run_path = local_bin / "april-run"
    command_run = shutil.which("run")
    command_path = Path(command_run) if command_run else None
    run_found = command_path is not None
    command_is_april = bool(command_path and is_april_wrapper(command_path))
    command_points_to_expected = bool(
        command_path and run_path.exists() and _same_file(command_path, run_path)
    )

    table = Table(title="APRIL Launcher Doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("APRIL_HOME", str(home))
    python_exists = (home / ".venv/bin/python").exists()
    table.add_row(".venv/bin/python exists", "yes" if python_exists else "no")
    table.add_row(f"{run_path} exists", "yes" if run_path.exists() else "no")
    table.add_row(f"{april_run_path} exists", "yes" if april_run_path.exists() else "no")
    table.add_row("run wrapper APRIL-owned", "yes" if is_april_wrapper(run_path) else "no")
    table.add_row(
        "april-run wrapper APRIL-owned",
        "yes" if is_april_wrapper(april_run_path) else "no",
    )
    table.add_row("run wrapper executable", "yes" if os.access(run_path, os.X_OK) else "no")
    table.add_row(
        "april-run wrapper executable",
        "yes" if os.access(april_run_path, os.X_OK) else "no",
    )
    table.add_row(f"{local_bin} in PATH", "yes" if path_contains_dir(local_bin) else "no")
    table.add_row("command -v run", command_run or "not found")
    table.add_row("command -v run is APRIL wrapper", "yes" if command_is_april else "no")
    table.add_row(
        "command -v run points to ~/.local/bin/run",
        "yes" if command_points_to_expected else "no",
    )
    console.print(table)
    _print_status(manager.status())

    if not run_found:
        console.print("[yellow]run was not found in PATH.[/yellow]")
        console.print(f"cd {home}")
        console.print("make install-global")
        console.print('export PATH="$HOME/.local/bin:$PATH"')
        console.print("run april --fake")
    elif not command_is_april:
        console.print("[yellow]run resolves to a non-APRIL command.[/yellow]")
        console.print(f"cd {home}")
        console.print("make install-global-force")
    elif path_contains_dir(local_bin):
        console.print("[green]OK: run resolves to an APRIL wrapper visible in PATH.[/green]")


def _print_verification_table(title: str, checks: list[VerifyCheck]) -> None:
    table = Table(title=title)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in checks:
        table.add_row(check.name, check.status or ("pass" if check.ok else "fail"), check.detail)
    console.print(table)


def _print_model_doctor(payload: dict[str, Any]) -> None:
    table = Table(title="APRIL Model Doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("Python", str(payload["python_version"]))
    table.add_row("APRIL_HOME", str(payload["april_home"]))
    table.add_row("Runtime backend", str(payload["runtime_backend"]))
    table.add_row(
        "llama-cpp-python installed",
        "yes" if payload["llama_cpp_python_installed"] else "no",
    )
    table.add_row("API token", str(payload["api_token"]))
    table.add_row("Runtime token", str(payload["runtime_token"]))
    table.add_row("Machine", str(payload["machine"]))
    table.add_row("CPU count", str(payload["cpu_count"]))
    table.add_row("Estimated RAM", str(payload["estimated_ram"]))
    console.print(table)

    models = Table(title="Configured Models")
    for column in (
        "ID",
        "Role",
        "Path",
        "Exists",
        "Size",
        "Ctx",
        "Threads",
        "Batch",
        "Keep",
        "Idle unload",
        "Realism",
    ):
        models.add_column(column)
    for model in payload["models"]:
        models.add_row(
            str(model["id"]),
            str(model["role"]),
            str(model["path"]),
            "yes" if model["path_exists"] else "no",
            str(model["file_size"]),
            str(model["context_size"]),
            str(model["threads"]),
            str(model["n_batch"] or "-"),
            "yes" if model["keep_loaded"] else "no",
            str(model["idle_unload_seconds"] or "-"),
            str(model["realism"]),
        )
    console.print(models)


def _print_model_recommendation(payload: dict[str, Any]) -> None:
    table = Table(title="APRIL Model Recommendation")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Architecture", str(payload["architecture"]))
    table.add_row("Platform", str(payload["platform"]))
    table.add_row("Python machine", str(payload["python_machine"]))
    table.add_row("arm64 Python", "yes" if payload["arm64_python"] else "no")
    table.add_row("CPU count", str(payload["cpu_count"]))
    table.add_row("Available memory", str(payload["available_memory"]))
    table.add_row("Recommended profile", str(payload["recommended_profile"]))
    table.add_row("Expected backend", str(payload["expected_backend"]))
    console.print(table)
    console.print("[bold]Notes[/bold]")
    for note in payload["notes"]:
        console.print(f"- {note}")
    console.print("[bold]Commands you may run manually[/bold]")
    for command in payload["manual_commands"]:
        console.print(f"  {command}")
    console.print(
        "[dim]This command only inspects local hardware. It does not install packages, "
        "download models, modify shell files, switch configuration, or send data.[/dim]"
    )


def _print_benchmark(results: list[BenchmarkResult]) -> None:
    table = Table(title="APRIL Model Benchmark")
    table.add_column("Run")
    table.add_column("Load")
    table.add_column("First token")
    table.add_column("Generation")
    table.add_column("Tokens")
    table.add_column("Tokens/sec")
    table.add_column("Unload")
    table.add_column("Detail")
    for result in results:
        table.add_row(
            str(result.run_index),
            f"{result.load_time_seconds:.2f}s",
            "n/a"
            if result.first_token_latency_seconds is None
            else f"{result.first_token_latency_seconds:.2f}s",
            f"{result.generation_time_seconds:.2f}s",
            str(result.output_tokens),
            f"{result.tokens_per_second:.2f}",
            "yes" if result.unload_success else "no",
            result.detail,
        )
    console.print(table)
    console.print(
        "CPU-only recommendation: keep contexts conservative, use small batch sizes, "
        "and unload non-brain models when not in active use."
    )


def _print_brain_eval(results: list[Any]) -> None:
    table = Table(title="APRIL Brain Eval")
    table.add_column("Case")
    table.add_column("Status")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Detail")
    for result in results:
        actual_intent = result.actual.get("intent", "-")
        actual_agent = result.actual.get("agent", "-")
        table.add_row(
            result.id,
            "pass" if result.ok else "fail",
            f"{result.expected_intent}/{result.expected_agent}",
            f"{actual_intent}/{actual_agent}",
            result.detail,
        )
    console.print(table)


@april_app.callback(invoke_without_command=True)
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
        _delegate(["chat"], fake=fake, oneshot=oneshot)


@april_app.command()
def chat(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["chat"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
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
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
def status(json_output: bool = typer.Option(False, "--json")) -> None:
    status_value = _manager().status()
    if json_output:
        console.print_json(data=_status_payload(status_value))
        return
    _print_status(status_value)


@app.command()
def doctor() -> None:
    _doctor()


@april_app.command("doctor")
def april_doctor() -> None:
    _doctor()


@april_app.command("readiness")
def readiness(json_output: bool = typer.Option(False, "--json")) -> None:
    """Explain offline exactly what is missing for real local-model readiness.

    Reads only configs/env; never starts a service, loads a model, opens the
    microphone, downloads anything, or installs anything. Prints actionable
    commands only. Paths and tokens are redacted.
    """
    report = build_readiness_report(_manager().home)
    if json_output:
        console.print_json(data=report.model_dump())
        return
    _print_readiness(report)


_READINESS_STATUS_STYLE = {
    "ok": "[green]ok[/green]",
    "warning": "[yellow]warning[/yellow]",
    "blocker": "[red]blocker[/red]",
    "skipped": "[dim]skipped[/dim]",
}


def _print_readiness(report: ReadinessReport) -> None:
    headline = (
        "[green]real model ready[/green]"
        if report.real_model_ready
        else "[red]NOT real-model ready[/red]"
    )
    console.print(
        f"APRIL readiness — {headline} (backend={report.runtime_backend}, env={report.environment})"
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
        console.print("[green]No blockers: run the real verification command to confirm.[/green]")


@april_app.command()
def health(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["health"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
def models(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["models"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
def briefing(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["briefing"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
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
    _ensure_services(_effective_fake(ctx, fake))
    manager = _manager()
    token = manager.settings.api.token
    base_url = _desktop_base_url(manager)
    if not token:
        console.print("[red]No API token configured. Run `run april setup tokens` first.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]APRIL Desktop is available at {base_url}[/green]")
    console.print("The API token is passed locally (URL fragment / JS bridge) and never logged.")
    if no_open:
        return
    if native:
        if _open_desktop_native(base_url, token):
            return
        console.print(
            "[yellow]pywebview is not installed. Install the optional native window with "
            "`pip install -e '.[desktop]'`, or use the default browser path.[/yellow]"
        )
    fragment_url = f"{base_url}#token={token}"
    if not _open_desktop_browser(fragment_url):
        console.print(
            f"[yellow]Could not open a browser automatically. Open {base_url} and append "
            "your token as #token=... in the address bar.[/yellow]"
        )


@april_app.command()
def approvals(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["approvals"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
def approve(
    ctx: typer.Context,
    approval_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["approve", approval_id], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx)
    )


@april_app.command()
def deny(
    ctx: typer.Context,
    approval_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["deny", approval_id], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx)
    )


@april_app.command()
def projects(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["projects"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@model_app.command("load")
def model_load(
    ctx: typer.Context,
    model_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["model", "load", model_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@model_app.command("unload")
def model_unload(
    ctx: typer.Context,
    model_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["model", "unload", model_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@model_app.command("doctor")
def model_doctor_command(json_output: bool = typer.Option(False, "--json")) -> None:
    payload = model_doctor(_manager().home)
    if json_output:
        console.print_json(data=payload)
        return
    _print_model_doctor(payload)


@model_app.command("recommend")
def model_recommend_command(json_output: bool = typer.Option(False, "--json")) -> None:
    """Report a non-mutating model-profile recommendation for this Mac.

    Inspects only local hardware. It never installs, downloads, switches
    configuration, edits shell files, or sends data anywhere.
    """
    payload = recommend_model_profile(_manager().home)
    if json_output:
        console.print_json(data=payload)
        return
    _print_model_recommendation(payload)


@model_app.command("import")
def model_import_command(
    role: str = typer.Option(..., "--role"),
    model_id: str = typer.Option(..., "--id"),
    name: str = typer.Option(..., "--name"),
    path: Path = typer.Option(..., "--path"),
    copy_into_models: bool = typer.Option(False, "--copy-into-models"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    try:
        result = import_model(
            home=_manager().home,
            role=role,
            model_id=model_id,
            name=name,
            source_path=path,
            copy_into_models=copy_into_models,
            force=force,
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Registered {result.model_id} for role {result.role}.[/green]")
    console.print(f"Model path: {result.path}")
    console.print(result.next_command)


@model_app.command("benchmark")
def model_benchmark_command(
    model_path: Path,
    prompt: str = typer.Option("Reply with one short sentence.", "--prompt"),
    runs: int = typer.Option(1, "--runs", min=1, max=20),
    max_output_tokens: int = typer.Option(32, "--max-output-tokens", min=1, max=4096),
    keep_loaded: bool = typer.Option(False, "--keep-loaded"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if not model_path.expanduser().exists():
        console.print(f"[red]GGUF path does not exist: {model_path}[/red]")
        raise typer.Exit(1)
    results = run_model_benchmark(
        _manager().home,
        model_path,
        prompt=prompt,
        runs=runs,
        max_output_tokens=max_output_tokens,
        keep_loaded=keep_loaded,
    )
    if json_output:
        console.print_json(data={"runs": [result.model_dump() for result in results]})
    else:
        _print_benchmark(results)
    if not all(result.ok for result in results):
        raise typer.Exit(1)


profile_app = typer.Typer(help="Model profile operations.")
model_app.add_typer(profile_app, name="profile")


@profile_app.command("list")
def model_profile_list() -> None:
    profiles = load_model_profiles(_manager().home)
    table = Table(title="APRIL Model Profiles")
    table.add_column("Profile")
    table.add_column("Description")
    for name, profile in profiles.items():
        description = profile.get("description", "") if isinstance(profile, dict) else ""
        table.add_row(str(name), str(description))
    console.print(table)


@profile_app.command("apply")
def model_profile_apply(profile_name: str) -> None:
    try:
        backup = apply_model_profile(home=_manager().home, profile_name=profile_name)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Applied model profile: {profile_name}[/green]")
    console.print(f"Backup: {backup}")


@project_app.command("add")
def project_add(
    ctx: typer.Context,
    path: str,
    name: str | None = typer.Option(None, "--name"),
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    args = ["project", "add", path]
    if name:
        args.extend(["--name", name])
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@project_app.command("index")
def project_index(
    ctx: typer.Context,
    project_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["project", "index", project_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@memory_app.command("list")
def memory_list(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["memory", "list"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@memory_app.command("search")
def memory_search(
    ctx: typer.Context,
    query: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["memory", "search", query],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@memory_app.command("delete")
def memory_delete(
    ctx: typer.Context,
    memory_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["memory", "delete", memory_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@memory_app.command("export")
def memory_export(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["memory", "export"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@memory_app.command("reindex")
def memory_reindex(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["memory", "reindex"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@memory_app.command("doctor")
def memory_doctor(
    json_output: bool = typer.Option(False, "--json"),
    verify_runtime_embedding: bool = typer.Option(
        False,
        "--verify-runtime-embedding",
        help="Explicitly call April Runtime /runtime/embed to verify local semantic embeddings.",
    ),
) -> None:
    settings = load_settings(root=_manager().home)
    data = _memory_doctor_report(settings, verify_runtime_embedding=verify_runtime_embedding)
    if json_output:
        console.print_json(data=data)
        return
    _print_memory_doctor(data)


def _memory_doctor_report(
    settings: Any, *, verify_runtime_embedding: bool = False
) -> dict[str, Any]:
    configured_provider = settings.memory.embedding_provider
    runtime_local_requested = configured_provider == "runtime-local"
    model_info = _embedding_model_info(settings)
    index = _vector_index_report(settings)
    verification: dict[str, Any] | None = None
    verified_dimensions: int | None = None
    verified_ok = False
    if verify_runtime_embedding and runtime_local_requested:
        verification = _verify_runtime_embedding(settings, model_info.get("model_id"))
        verified_ok = verification.get("status") == "ok"
        raw_dimensions = verification.get("dimensions")
        verified_dimensions = raw_dimensions if type(raw_dimensions) is int else None

    model_ready = bool(
        model_info["embedding_model_registered"] and model_info["embedding_model_path_exists"]
    )
    fallback_to_hashed = runtime_local_requested and (
        not model_ready or (verification is not None and not verified_ok)
    )
    active_provider = "hashed-token" if fallback_to_hashed else configured_provider
    if active_provider == "hashed-token":
        active_dimensions: int | None = HashedTokenEmbedding().dimensions
    else:
        active_dimensions = verified_dimensions or index.get("persisted_dimensions")

    persisted_provider = index.get("persisted_provider")
    persisted_dimensions = index.get("persisted_dimensions")
    reindex_required = False
    if (persisted_provider is not None and persisted_provider != active_provider) or (
        persisted_dimensions is not None
        and active_dimensions is not None
        and persisted_dimensions != active_dimensions
    ):
        reindex_required = True

    if reindex_required:
        status = "reindex_required"
    elif (runtime_local_requested and not model_ready) or (
        verification is not None and not verified_ok
    ):
        status = "not_ready"
    elif runtime_local_requested and verify_runtime_embedding and verified_ok:
        status = "ok"
    elif runtime_local_requested:
        status = "configured_unverified"
    else:
        status = "ok"

    report: dict[str, Any] = {
        "status": status,
        "configured_embedding_provider": configured_provider,
        "active_vector_index_provider": active_provider,
        "dimensions": active_dimensions,
        "runtime_local_requested": runtime_local_requested,
        "fell_back_to_hashed_token": fallback_to_hashed,
        "fallback_risk": runtime_local_requested and not verified_ok,
        "reindex_required": reindex_required,
        "embedding_model_id": model_info.get("model_id"),
        "embedding_role_model_registered": model_info["embedding_model_registered"],
        "embedding_model_path_exists": model_info["embedding_model_path_exists"],
        "embedding_model_path_basename": model_info.get("path_basename"),
        "vector_index": index,
    }
    if verification is not None:
        report["runtime_embedding_verification"] = verification
    return report


def _embedding_model_info(settings: Any) -> dict[str, Any]:
    try:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml", root=settings.home
        )
    except ConfigError as exc:
        return {
            "embedding_model_registered": False,
            "embedding_model_path_exists": False,
            "model_id": settings.memory.embedding_model_id,
            "path_basename": None,
            "registry_error": str(exc),
        }
    candidates = [model for model in registry.list() if model.role == "embedding"]
    selected = None
    if settings.memory.embedding_model_id:
        for model in candidates:
            if model.id == settings.memory.embedding_model_id:
                selected = model
                break
    elif candidates:
        selected = candidates[0]
    if selected is None:
        return {
            "embedding_model_registered": False,
            "embedding_model_path_exists": False,
            "model_id": settings.memory.embedding_model_id,
            "path_basename": None,
        }
    resolved = selected.resolved_path(settings.home)
    return {
        "embedding_model_registered": True,
        "embedding_model_path_exists": resolved.exists(),
        "model_id": selected.id,
        "path_basename": resolved.name,
    }


def _vector_index_report(settings: Any) -> dict[str, Any]:
    metadata_path = settings.vector_index_path / "metadata.json"
    persisted_provider: str | None = None
    persisted_dimensions: int | None = None
    record_count = 0
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if isinstance(metadata, dict):
            raw_provider = metadata.get("provider")
            raw_dimensions = metadata.get("dimensions")
            raw_records = metadata.get("record_count")
            persisted_provider = raw_provider if isinstance(raw_provider, str) else None
            persisted_dimensions = raw_dimensions if type(raw_dimensions) is int else None
            record_count = raw_records if type(raw_records) is int and raw_records >= 0 else 0
    return {
        "path_basename": settings.vector_index_path.name,
        "persisted_provider": persisted_provider,
        "persisted_dimensions": persisted_dimensions,
        "record_count": record_count,
    }


def _verify_runtime_embedding(settings: Any, model_id: str | None) -> dict[str, Any]:
    client = RuntimeClient(
        settings.runtime.url,
        timeout=settings.runtime.request_timeout_seconds,
        token=settings.runtime.token,
    )

    async def _probe() -> list[float]:
        return await client.embed("april memory doctor", model_id=model_id)

    try:
        vector = asyncio.run(_probe())
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:240]}
    return {
        "status": "ok",
        "model_id": model_id,
        "dimensions": len(vector),
    }


def _print_memory_doctor(data: dict[str, Any]) -> None:
    table = Table(title="APRIL Memory Doctor")
    table.add_column("Field")
    table.add_column("Value")
    for key in (
        "status",
        "configured_embedding_provider",
        "active_vector_index_provider",
        "dimensions",
        "runtime_local_requested",
        "fell_back_to_hashed_token",
        "fallback_risk",
        "reindex_required",
        "embedding_model_id",
        "embedding_role_model_registered",
        "embedding_model_path_exists",
        "embedding_model_path_basename",
    ):
        table.add_row(key, str(data.get(key)))
    console.print(table)


@conversation_app.command("delete")
def conversation_delete(
    ctx: typer.Context,
    conversation_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["conversation", "delete", conversation_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@reminder_app.command("list")
def reminder_list(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["reminder", "list"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@reminder_app.command("create")
def reminder_create(
    ctx: typer.Context,
    content: str,
    due_at: str | None = typer.Option(None, "--due-at"),
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    args = ["reminder", "create", content]
    if due_at:
        args.extend(["--due-at", due_at])
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@reminder_app.command("delete")
def reminder_delete(
    ctx: typer.Context,
    reminder_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["reminder", "delete", reminder_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@task_app.command("list")
def task_list(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["task", "list"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@eval_app.command("brain")
def eval_brain(
    fake: bool = typer.Option(False, "--fake", help="Run deterministic fake Brain eval."),
    real_model: Path | None = typer.Option(None, "--real-model"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if fake:
        results = run_fake_brain_eval(_manager().home)
    elif real_model is not None:
        if not real_model.expanduser().exists():
            console.print(f"[red]GGUF path does not exist: {real_model}[/red]")
            raise typer.Exit(1)
        results = run_real_brain_eval(_manager().home, real_model)
    else:
        console.print("[red]Use --fake or --real-model /path/to/model.gguf.[/red]")
        raise typer.Exit(1)
    if json_output:
        console.print_json(data={"results": [result.model_dump() for result in results]})
    else:
        _print_brain_eval(results)
    if not all(result.ok for result in results):
        raise typer.Exit(1)


@voice_app.command("health")
def voice_health(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "health"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("doctor")
def voice_doctor(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "doctor"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("verify-live")
def voice_verify_live(
    report: Path | None = typer.Option(
        None, "--report", help="Write a redacted live voice verification report JSON here."
    ),
    seconds: float = typer.Option(3.0, "--seconds", min=0.2, max=10.0),
    retain_debug_audio: bool = typer.Option(
        False,
        "--retain-debug-audio",
        help="Keep the exact temporary audio files created by this explicit verification run.",
    ),
) -> None:
    settings = _manager().settings
    doctor = collect_voice_doctor(settings)
    console.print(f"Voice doctor status: {doctor['status']}")
    guidance = doctor.get("macos_microphone_permission_guidance")
    if guidance:
        console.print(str(guidance))
    console.print("Wake-word listening is not used by this verification.")

    def confirm(message: str) -> bool:
        return typer.confirm(message, default=False)

    result = asyncio.run(
        run_voice_live_verification(
            settings=settings,
            confirm_recording=confirm,
            confirm_transcription=confirm,
            confirm_playback=confirm,
            seconds=seconds,
            retain_debug_audio=retain_debug_audio,
            report_path=report,
        )
    )
    console.print(
        "Voice live verification: "
        f"{result.summary} (recording={result.recording_success}, "
        f"stt={result.stt_success}, transcript_length={result.transcript_length}, "
        f"tts={result.tts_success}, playback_confirmed={result.playback_user_confirmed})"
    )
    if report is not None:
        console.print(f"[green]Wrote voice verification report to {report.expanduser()}[/green]")
    if result.summary != "pass":
        raise typer.Exit(1)


@voice_app.command("devices")
def voice_devices(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "devices"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("ptt")
def voice_ptt(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    seconds: float | None = typer.Option(None, "--seconds", min=0.1, max=300.0),
) -> None:
    args = ["voice", "ptt"]
    if seconds is not None:
        args.extend(["--seconds", str(seconds)])
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@voice_app.command("test-record")
def voice_test_record(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    seconds: float = typer.Option(3.0, "--seconds", min=0.1, max=30.0),
) -> None:
    _delegate(
        ["voice", "test-record", "--seconds", str(seconds)],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("test-stt")
def voice_test_stt(
    ctx: typer.Context,
    audio_path: Path,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "test-stt", str(audio_path)],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("test-tts")
def voice_test_tts(
    ctx: typer.Context,
    text: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "test-tts", text],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("listen")
def voice_listen(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "listen"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@agent_app.command("run")
def agent_run(
    ctx: typer.Context,
    agent: str,
    message: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    project_id: str | None = typer.Option(None, "--project-id"),
    repo_path: str | None = typer.Option(None, "--repo-path"),
    conversation_id: str | None = typer.Option(None, "--conversation-id"),
) -> None:
    args = ["agent", "run", agent, message]
    if project_id:
        args.extend(["--project-id", project_id])
    if repo_path:
        args.extend(["--repo-path", repo_path])
    if conversation_id:
        args.extend(["--conversation-id", conversation_id])
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@config_app.command("validate")
def config_validate() -> None:
    errors = validate_configuration(_manager().home)
    if errors:
        console.print("[red]APRIL configuration is invalid.[/red]")
        for error in errors:
            console.print(f"- {error}")
        raise typer.Exit(1)
    console.print("[green]APRIL configuration is valid.[/green]")


@config_app.command("inspect")
def config_inspect() -> None:
    errors = validate_configuration(_manager().home)
    if errors:
        console.print("[red]APRIL configuration is invalid.[/red]")
        for error in errors:
            console.print(f"- {error}")
        raise typer.Exit(1)
    settings = load_settings(root=_manager().home)
    home = _manager().home
    settings_data = settings.model_dump(mode="json")
    if isinstance(settings_data.get("api"), dict):
        settings_data["api"]["token"] = "[REDACTED]"
    if isinstance(settings_data.get("runtime"), dict):
        settings_data["runtime"]["token"] = "[REDACTED]"
    models = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    data = {
        "settings": settings_data,
        "models": [model.model_dump(mode="json") for model in models.list()],
        "agents": load_agents_file(home).model_dump(mode="json"),
        "tools": load_tools_file(home).model_dump(mode="json"),
        "permissions": load_permissions_file(home).model_dump(mode="json"),
    }
    console.print_json(data=data)


def _run_profile_op(operation: Callable[[UserProfileStore], Awaitable[_T]]) -> _T:
    async def _run() -> _T:
        settings = load_settings(root=_manager().home)

        async with Database(settings.database_path) as database:
            await run_migrations(database)
            return await operation(UserProfileStore(database))

    return asyncio.run(_run())


@user_profile_app.command("show")
def profile_show() -> None:
    """Inspect the local user profile. It is stored only on this machine."""
    profile = _run_profile_op(lambda store: store.get())
    if profile is None:
        console.print("No local profile is set. Use `run april profile set --display-name ...`.")
        return
    console.print_json(data=profile.model_dump())


@user_profile_app.command("set")
def profile_set(
    display_name: str = typer.Option(..., "--display-name"),
    address: str | None = typer.Option(
        None, "--address", help="Preferred form of address (e.g. a first name)."
    ),
    timezone: str | None = typer.Option(None, "--timezone"),
) -> None:
    """Create or update the local user profile (explicit fields only)."""
    profile = _run_profile_op(
        lambda store: store.set(
            display_name=display_name, preferred_address=address, timezone=timezone
        )
    )
    console.print_json(data=profile.model_dump())


@user_profile_app.command("delete")
def profile_delete() -> None:
    """Delete the local user profile."""
    deleted = _run_profile_op(lambda store: store.delete())
    console.print(f"Deleted local profile: {deleted}")


@setup_app.command("models")
def setup_models(
    brain: Path | None = typer.Option(None, "--brain", help="Local brain GGUF path."),
    coding: Path | None = typer.Option(None, "--coding", help="Local coding GGUF path."),
    reading: Path | None = typer.Option(None, "--reading", help="Local reading GGUF path."),
    reasoning: Path | None = typer.Option(
        None, "--reasoning", help="Optional reasoning GGUF path."
    ),
    brain_id: str | None = typer.Option(None, "--brain-id"),
    coding_id: str | None = typer.Option(None, "--coding-id"),
    reading_id: str | None = typer.Option(None, "--reading-id"),
    reasoning_id: str | None = typer.Option(None, "--reasoning-id"),
    copy_into_models: bool = typer.Option(False, "--copy-into-models"),
    apply_changes: bool = typer.Option(False, "--apply"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Safely validate and optionally configure APRIL's local GGUF model set."""
    if apply_changes and dry_run:
        console.print("[red]Use either --apply or --dry-run, not both.[/red]")
        raise typer.Exit(1)
    try:
        result = setup_model_set(
            home=_manager().home,
            role_paths={
                "brain": brain,
                "coding": coding,
                "reading": reading,
                "reasoning": reasoning,
            },
            role_ids={
                "brain": brain_id,
                "coding": coding_id,
                "reading": reading_id,
                "reasoning": reasoning_id,
            },
            copy_into_models=copy_into_models,
            apply=apply_changes,
            force=force,
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        "[green]Model setup applied.[/green]"
        if result["applied"]
        else "[yellow]Model setup dry run; no files were changed.[/yellow]"
    )
    for entry in result["entries"]:
        console.print(
            f"{entry['role']}: {entry['source_basename']} -> {entry['model_id']} "
            f"(copy_into_models={entry['copy_into_models']})"
        )
    if result["backup_basename"]:
        console.print(f"Config backup: {result['backup_basename']}")
    console.print("Next commands:")
    for command in result["next_commands"]:
        console.print(f"  {command}")


@setup_app.command("voice")
def setup_voice(
    whisper_binary: Path = typer.Option(..., "--whisper-binary"),
    whisper_model: Path = typer.Option(..., "--whisper-model"),
    piper_binary: Path = typer.Option(..., "--piper-binary"),
    piper_model: Path = typer.Option(..., "--piper-model"),
    wake_word_model: Path | None = typer.Option(None, "--wake-word-model"),
    apply_changes: bool = typer.Option(False, "--apply"),
    enable: bool = typer.Option(
        False,
        "--enable",
        help="Turn voice ON after required paths validate. Voice stays OFF without this flag.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Validate and optionally configure local voice tools without recording."""
    if apply_changes and dry_run:
        console.print("[red]Use either --apply or --dry-run, not both.[/red]")
        raise typer.Exit(1)
    try:
        result = setup_voice_stack(
            home=_manager().home,
            whisper_binary=whisper_binary,
            whisper_model=whisper_model,
            piper_binary=piper_binary,
            piper_model=piper_model,
            wake_word_model=wake_word_model,
            apply=apply_changes,
            enable=enable,
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        "[green]Voice setup applied.[/green]"
        if result["applied"]
        else "[yellow]Voice setup dry run; no files were changed.[/yellow]"
    )
    for artifact in result["artifacts"]:
        label = artifact["basename"] or "not configured"
        console.print(f"{artifact['name']}: {label}")
    for warning in result["warnings"]:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    # Voice is never enabled by surprise: state the enabled/disabled outcome plainly.
    if result["voice_enabled"]:
        console.print("[green]Voice is now ENABLED.[/green]")
        if result["wake_word_available"]:
            console.print(
                "Push-to-talk is available. Wake-word listening stays UNVERIFIED until "
                "`run april voice verify-live` passes on this Mac."
            )
        else:
            console.print(
                "Push-to-talk is available. No wake-word model is configured, so wake-word "
                "listening is UNAVAILABLE; push-to-talk works without one."
            )
    elif apply_changes and enable:
        # enable was requested but apply did not run (should not happen, but be honest).
        console.print("[yellow]Voice remains DISABLED.[/yellow]")
    else:
        console.print(
            "[yellow]Voice remains DISABLED.[/yellow] Paths are validated only; voice stays OFF "
            "unless you re-run with --apply --enable."
        )
    if result["backup_basename"]:
        console.print(f"Config backup: {result['backup_basename']}")
    console.print("Next commands:")
    for command in result["next_commands"]:
        console.print(f"  {command}")


@setup_app.command("app-stub")
def setup_app_stub(
    output: Path = typer.Option(Path("dist/APRIL.app"), "--output"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Create the unsigned local-development macOS APRIL.app launcher."""
    try:
        result = create_macos_app_stub(home=_manager().home, output=output, force=force)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Created unsigned APRIL development launcher: {result.output_path}[/green]"
    )
    console.print("Unsigned local development only. No models, tokens, signing, or notarization.")


@setup_app.command("tokens")
def setup_tokens(
    output: Path = typer.Option(Path(".env"), "--output", help="Local env file to update."),
) -> None:
    target = output if output.is_absolute() else _manager().home / output
    write_token_env_file(target, generate_tokens())
    console.print(f"Generated APRIL API and Runtime tokens in {target}.")
    console.print("Full token values were not printed.")


@setup_app.command("bootstrap")
def setup_bootstrap(
    env_file: Path = typer.Option(Path(".env"), "--env-file", help="Local env file for tokens."),
    force: bool = typer.Option(False, "--force", help="Regenerate tokens even if they exist."),
    apply_profile: bool = typer.Option(
        False, "--apply-profile", help="Apply the recommended model profile (mutates configs)."
    ),
    show_paths: bool = typer.Option(
        False, "--show-paths", help="Include absolute local paths in bootstrap output."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Safe, non-destructive local first-run setup. Never prints full tokens."""
    home = _manager().home
    target_env = env_file if env_file.is_absolute() else home / env_file
    report = run_bootstrap(
        home,
        env_file=target_env,
        force=force,
        apply_profile=apply_profile,
        show_paths=show_paths,
    )
    if json_output:
        console.print_json(data=report)
    else:
        _print_bootstrap(report)
    if not report["config_valid"]:
        raise typer.Exit(1)


def _print_bootstrap(report: dict[str, Any]) -> None:
    console.print(f"[bold]APRIL bootstrap[/bold] — home: {report['home']}")
    created = sum(1 for item in report["directories"] if item["created"])
    console.print(f"Directories: {len(report['directories'])} ensured ({created} newly created).")
    tokens = report["tokens"]
    console.print(f"Tokens ({report['env_file']}): {tokens['action']} (values not printed).")
    for warning in report["dev_token_warnings"]:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    machine = report["machine"]
    console.print(
        f"Machine: {machine['architecture']} · {machine['cpu_count']} CPUs · "
        f"{machine['available_memory']} RAM"
    )
    console.print(
        f"Recommended profile: {report['recommended_profile']} "
        f"({report['expected_backend']}); "
        + (
            f"applied {report['applied_profile']}."
            if report["profile_applied"]
            else "not applied (use --apply-profile)."
        )
    )
    console.print(
        f"llama-cpp-python: {'available' if report['llama_cpp_available'] else 'not installed'}; "
        f"models missing files: {len(report['missing_model_paths'])}."
    )
    console.print(f"Allowed filesystem roots: {report['allowed_filesystem_roots']}")
    console.print(f"Config valid: {report['config_valid']}")
    console.print("Next commands:")
    for command in report["next_commands"]:
        console.print(f"  {command}")


@april_app.command()
def verify(
    model_path: Path | None = typer.Argument(None),
    fake: bool = typer.Option(False, "--fake", help="Run deterministic fake-backend verification."),
    real_model: bool = typer.Option(False, "--real-model"),
    workflow: bool = typer.Option(False, "--workflow"),
    target_mac: bool = typer.Option(False, "--target-mac"),
    all_configured_models: bool = typer.Option(
        False,
        "--all-configured-models",
        "--mac-readiness",
        help="Verify every configured real GGUF model (load/chat/stream/unload + switching).",
    ),
    soak: bool = typer.Option(False, "--soak", help="Run a bounded fake-backend soak check."),
    minutes: float = typer.Option(10.0, "--minutes", min=0.01, max=240.0),
    soak_interval_seconds: float = typer.Option(
        1.0,
        "--soak-interval-seconds",
        min=0.1,
        max=60.0,
        help="Delay between fake soak iterations.",
    ),
    cycle_fake_models: bool = typer.Option(False, "--cycle-fake-models"),
    require_real_model: bool = typer.Option(False, "--require-real-model"),
    json_output: bool = typer.Option(False, "--json"),
    report: Path | None = typer.Option(
        None, "--report", help="Write a redacted machine-readable verification report JSON here."
    ),
    min_tokens_per_second: float | None = typer.Option(None, "--min-tokens-per-second", min=0.0),
    max_load_seconds: float | None = typer.Option(None, "--max-load-seconds", min=0.0),
    max_first_token_latency_seconds: float | None = typer.Option(
        None, "--max-first-token-latency-seconds", min=0.0
    ),
    max_rss_mb: float | None = typer.Option(None, "--max-rss-mb", min=0.0),
    min_routing_accuracy: float = typer.Option(0.90, "--min-routing-accuracy", min=0.0, max=1.0),
    max_output_tokens: int = typer.Option(32, "--max-output-tokens", min=1, max=4096),
    timeout: float = typer.Option(180.0, "--timeout", min=1.0),
) -> None:
    thresholds = ReportThresholds(
        min_tokens_per_second=min_tokens_per_second,
        max_load_seconds=max_load_seconds,
        max_first_token_latency_seconds=max_first_token_latency_seconds,
        max_rss_mb=max_rss_mb,
        min_routing_accuracy=min_routing_accuracy,
    )
    if soak:
        soak_report = run_fake_soak(
            _manager().home,
            minutes=minutes,
            interval_seconds=soak_interval_seconds,
            cycle_models=cycle_fake_models,
        )
        checks = [
            VerifyCheck(
                name="fake soak",
                ok=soak_report.summary == "pass",
                detail=f"iterations={soak_report.iterations}, failures={len(soak_report.failures)}",
            )
        ]
        if json_output:
            console.print_json(data=soak_report.model_dump())
        else:
            _print_verification_table("APRIL Fake Soak Verification", checks)
        if report is not None:
            written = write_soak_report(soak_report, report)
            console.print(
                f"[green]Wrote fake soak report to {written}[/green] "
                f"(summary: {soak_report.summary}, real_model_verified: false)"
            )
        if soak_report.summary != "pass":
            raise typer.Exit(1)
        raise typer.Exit(0)
    if all_configured_models:
        verifier = run_all_configured_models_verification(
            _manager().home,
            require_real_model=require_real_model,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            thresholds=thresholds,
        )
        checks = verifier.checks
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _print_verification_table("APRIL All-Configured-Model Verification", checks)
        if report is not None:
            multi_report = verifier.build_report()
            written = write_multi_model_report(multi_report, report)
            console.print(
                f"[green]Wrote multi-model verification report to {written}[/green] "
                f"(summary: {multi_report.summary}, "
                f"verification_level: {multi_report.verification_level})"
            )
        if not all(check.ok for check in checks):
            raise typer.Exit(1)
        raise typer.Exit(0)
    if target_mac:
        validator = TargetMacValidator(
            home=_manager().home,
            model_path=model_path,
            require_real_model=require_real_model,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        checks = validator.run()
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _print_verification_table("APRIL Target Mac Validation", checks)
        if report is not None:
            rendered = validator.build_report(thresholds=thresholds)
            written = write_report(rendered, report)
            console.print(
                f"[green]Wrote verification report to {written}[/green] "
                f"(summary: {rendered.summary})"
            )
        if not all(check.ok for check in checks):
            raise typer.Exit(1)
        raise typer.Exit(0)
    if workflow:
        checks = run_workflow_verification(
            _manager().home,
            real_model=real_model,
            model_path=model_path,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _print_verification_table("APRIL Workflow Verification", checks)
        if report is not None:
            workflow_report = build_workflow_report(
                checks,
                real_model_requested=real_model,
                timeout_seconds=timeout,
                max_output_tokens=max_output_tokens,
            )
            written = write_workflow_report(workflow_report, report)
            console.print(
                f"[green]Wrote workflow verification report to {written}[/green] "
                f"(summary: {workflow_report.summary}, "
                f"real_model_verified: {str(workflow_report.real_model_verified).lower()})"
            )
        if not all(check.ok for check in checks):
            raise typer.Exit(1)
        raise typer.Exit(0)
    if real_model:
        configured_path = model_path or (
            Path(os.environ["APRIL_TEST_GGUF_PATH"])
            if os.environ.get("APRIL_TEST_GGUF_PATH")
            else None
        )
        if configured_path is None:
            console.print(
                "[yellow]Skipping real-model verification: no GGUF path provided.[/yellow]"
            )
            raise typer.Exit(0)
        if not configured_path.expanduser().exists():
            console.print(f"[red]GGUF path does not exist: {configured_path}[/red]")
            raise typer.Exit(1)
        checks = run_real_model_verification(
            _manager().home,
            configured_path,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _print_verification_table("APRIL Real Model Verification", checks)
        if not all(check.ok for check in checks):
            raise typer.Exit(1)
        raise typer.Exit(0)
    if not fake:
        console.print("[red]Use --fake for deterministic local verification.[/red]")
        raise typer.Exit(1)
    checks = run_fake_verification(_manager().home)
    if json_output:
        console.print_json(data={"checks": [asdict(check) for check in checks]})
    else:
        _print_verification_table("APRIL Verification", checks)
    if not all(check.ok for check in checks):
        raise typer.Exit(1)


@april_app.command()
def stop() -> None:
    _print_status(_manager().stop())


@april_app.command()
def restart(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start services with fake runtime."),
) -> None:
    _print_status(_manager().restart(fake_backend=_effective_fake(ctx, fake)))


@april_app.command()
def logs(
    lines: int = typer.Option(80, "--lines", min=1, max=1000),
    tail: int | None = typer.Option(None, "--tail", min=1, max=1000),
) -> None:
    console.print(_manager().logs(lines=tail or lines))


if __name__ == "__main__":
    app()
