from __future__ import annotations

import json
import os
import shutil
import sys
import webbrowser
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.commands.composition import composition as _composition
from apps.runner.install import is_april_wrapper, path_contains_dir
from apps.runner.service_manager import AprilServiceManager, ServiceStatus
from apps.runner.verify import (
    BenchmarkResult,
    VerifyCheck,
)
from apps.runner.wake_live import run_sentinel_live_verification
from april_common.errors import ConfigError
from april_common.process_environment import ProcessCategory
from april_common.process_runner import run_restricted_process_sync
from april_common.settings import load_settings

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


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
    home = _composition._manager().home
    completed = run_restricted_process_sync(
        [sys.executable, "-m", "apps.cli.main", *args],
        cwd=home,
        category=ProcessCategory.CLI,
        timeout_seconds=24 * 60 * 60,
        max_stdout_bytes=10_000_000,
        max_stderr_bytes=10_000_000,
        april_home=home,
    )
    return completed.returncode if completed.returncode is not None else 1


def _ensure_services(fake: bool) -> ServiceStatus:
    try:
        status = _composition._manager().start(fake_backend=fake)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not status.ok:
        console.print("[red]APRIL services are not healthy.[/red]")
        _composition._print_status(status)
        raise typer.Exit(1)
    return status


def _delegate(args: list[str], *, fake: bool, oneshot: bool = False) -> None:
    manager = _composition._manager()
    before = manager.status()
    try:
        _composition._ensure_services(fake)
        if oneshot:
            console.print(
                "[yellow]APRIL oneshot mode: services will stop after this command.[/yellow]"
            )
        else:
            console.print("[green]APRIL services are running and will remain running.[/green]")
        code = _composition._run_april_cli(args)
    finally:
        if oneshot and not before.ok:
            console.print("[yellow]Stopping APRIL services started for oneshot mode.[/yellow]")
            _composition._print_status(manager.stop())
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
    manager = _composition._manager()
    home = manager.home
    from april_common.process_environment import PROCESS_ENVIRONMENT_POLICY_VERSION
    from services.evolution.adapters import inspect_adapter_state
    from services.jobs.worker import JOB_WORKER_STATUS_VERSION
    from services.tool_worker.limits import (
        UnsafeToolWorkerSocket,
        default_tool_worker_runtime_directory,
        validate_live_socket,
    )

    try:
        active_settings = load_settings(root=home)
        adapter_state = inspect_adapter_state(active_settings)
        adapter_state_label = (
            "consistent" if adapter_state["consistent"] else "reconciliation required"
        )
    except ConfigError:
        active_settings = None
        adapter_state_label = "unavailable (configuration invalid)"
    local_bin = Path.home() / ".local" / "bin"
    run_path = local_bin / "run"
    april_run_path = local_bin / "april-run"
    command_run = shutil.which("run")
    command_path = Path(command_run) if command_run else None
    run_found = command_path is not None
    command_is_april = bool(command_path and is_april_wrapper(command_path))
    command_points_to_expected = bool(
        command_path and run_path.exists() and _composition._same_file(command_path, run_path)
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
    table.add_row(
        "adapter pointer/database state",
        adapter_state_label,
    )
    table.add_row("child environment policy", PROCESS_ENVIRONMENT_POLICY_VERSION)
    if active_settings is not None and not active_settings.workers.tool_worker_enabled:
        table.add_row("Tool Worker", "disabled explicitly; risky tools fail closed")
    elif active_settings is not None:
        runtime_directory = default_tool_worker_runtime_directory(active_settings.home)
        try:
            mode = validate_live_socket(
                runtime_directory / "worker.sock",
                runtime_directory=runtime_directory,
            )
            table.add_row("Tool Worker", f"socket ready ({mode}); check /readiness for protocol")
        except FileNotFoundError:
            table.add_row("Tool Worker", "not running; start APRIL services")
        except UnsafeToolWorkerSocket as exc:
            table.add_row("Tool Worker", f"unsafe socket path ({exc})")
    if active_settings is not None and not active_settings.workers.job_worker_enabled:
        table.add_row("Job Worker", "disabled explicitly; durable jobs remain queued")
    elif active_settings is not None:
        status_path = active_settings.home / "data" / "runtime" / "job-worker" / "status.json"
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            valid = (
                payload.get("version") == JOB_WORKER_STATUS_VERSION and payload.get("ready") is True
            )
            table.add_row(
                "Job Worker",
                "ready" if valid else "status invalid or not ready; restart APRIL services",
            )
        except (OSError, json.JSONDecodeError):
            table.add_row("Job Worker", "not running; start APRIL services")
    console.print(table)
    _composition._print_status(manager.status())

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
    table.add_row("APRIL home", str(payload["april_home_basename"]))
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
