from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.install import is_april_wrapper, path_contains_dir
from apps.runner.service_manager import AprilServiceManager, ServiceStatus

app = typer.Typer(help="Global command dispatcher.")
april_app = typer.Typer(help="Run APRIL from any folder.", invoke_without_command=True)
app.add_typer(april_app, name="april")


def _manager() -> AprilServiceManager:
    return AprilServiceManager()


def _effective_fake(ctx: typer.Context, explicit: bool) -> bool:
    inherited = bool((ctx.obj or {}).get("fake", False))
    return inherited or explicit


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


def _ensure_services(fake: bool) -> None:
    try:
        status = _manager().start(fake_backend=fake)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not status.ok:
        console.print("[red]APRIL services are not healthy.[/red]")
        _print_status(status)
        raise typer.Exit(1)


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


@april_app.callback(invoke_without_command=True)
def april(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    ctx.obj = {"fake": fake}
    if ctx.invoked_subcommand is None:
        _ensure_services(fake)
        raise typer.Exit(_run_april_cli(["chat"]))


@april_app.command()
def chat(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _ensure_services(_effective_fake(ctx, fake))
    raise typer.Exit(_run_april_cli(["chat"]))


@april_app.command()
def ask(
    ctx: typer.Context,
    message: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    project_id: str | None = typer.Option(None, "--project-id"),
    repo_path: str | None = typer.Option(None, "--repo-path"),
) -> None:
    _ensure_services(_effective_fake(ctx, fake))
    args = ["ask", message]
    if project_id:
        args.extend(["--project-id", project_id])
    if repo_path:
        args.extend(["--repo-path", repo_path])
    raise typer.Exit(_run_april_cli(args))


@april_app.command()
def status() -> None:
    _print_status(_manager().status())


@app.command()
def doctor() -> None:
    _doctor()


@april_app.command("doctor")
def april_doctor() -> None:
    _doctor()


@april_app.command()
def health(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _ensure_services(_effective_fake(ctx, fake))
    raise typer.Exit(_run_april_cli(["health"]))


@april_app.command()
def models(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _ensure_services(_effective_fake(ctx, fake))
    raise typer.Exit(_run_april_cli(["models"]))


@april_app.command()
def approvals(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _ensure_services(_effective_fake(ctx, fake))
    raise typer.Exit(_run_april_cli(["approvals"]))


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
def logs(lines: int = typer.Option(80, "--lines", min=1, max=1000)) -> None:
    console.print(_manager().logs(lines=lines))


if __name__ == "__main__":
    app()
