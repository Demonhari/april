from __future__ import annotations

from pathlib import Path

import typer

from apps.cli.render import console
from apps.daemon.apriald import read_daemon_status, stop_daemon
from apps.runner.service_manager import AprilServiceManager
from april_common.audit import audit_logger_for_settings
from april_common.errors import ConfigError
from april_common.settings import load_settings
from services.memory.maintenance import (
    BackupCancelled,
    DatabaseMaintenanceError,
    check_database,
    create_backup,
    restore_backup,
)

database_app = typer.Typer(help="SQLite integrity, backup, and restore operations.")


@database_app.command("check")
def database_check(
    full: bool = typer.Option(False, "--full"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        settings = load_settings()
        result = check_database(
            settings.database_path,
            home=settings.home,
            full=full,
        )
    except ConfigError as exc:
        if json_output:
            console.print_json(data={"ok": False, "failures": ["configuration_unavailable"]})
        else:
            console.print("[red]Database check unavailable: configuration is invalid.[/red]")
        raise typer.Exit(2) from exc
    if json_output:
        console.print_json(data=result.to_dict())
    else:
        console.print(
            f"Database check: {'ok' if result.ok else 'failed'}; "
            f"quick_check={result.quick_check}; "
            f"foreign_keys={'ok' if result.foreign_key_consistent else 'failed'}; "
            f"journal={result.journal_mode}; schema={result.schema_version}."
        )
        if result.integrity_check is not None:
            console.print(f"Full integrity_check={result.integrity_check}.")
        if result.last_successful_backup:
            console.print(
                "Last successful backup: "
                f"{result.last_successful_backup.get('creation_timestamp', 'unknown')}."
            )
        for failure in result.failures:
            console.print(f"  failure: {failure}")
    if not result.ok:
        raise typer.Exit(1)


@database_app.command("backup")
def database_backup(
    output: Path = typer.Option(..., "--output"),
) -> None:
    try:
        settings = load_settings()
        result = create_backup(
            settings.database_path,
            output,
            home=settings.home,
            audit=audit_logger_for_settings(settings),
        )
    except BackupCancelled as exc:
        console.print("[yellow]Database backup cancelled; no output was published.[/yellow]")
        raise typer.Exit(130) from exc
    except (ConfigError, DatabaseMaintenanceError, OSError) as exc:
        console.print(
            f"[red]Database backup failed ({type(exc).__name__}); "
            "no incomplete backup was published.[/red]"
        )
        raise typer.Exit(1) from exc
    console.print(
        f"Database backup created: {result.output}; "
        f"schema={result.manifest.schema_version}; size={result.manifest.size}."
    )


@database_app.command("restore")
def database_restore(
    input_path: Path = typer.Option(..., "--input"),
    stop_services: bool = typer.Option(False, "--stop-services"),
) -> None:
    try:
        settings = load_settings()
        manager = AprilServiceManager(home=settings.home)
        status = manager.status()
        daemon_status = read_daemon_status(settings)
        daemon_running = isinstance(daemon_status.get("pid"), int)
        running = status.runtime.running or status.api.running or daemon_running
        if running and stop_services:
            if daemon_running:
                stopped_daemon = stop_daemon(settings)
                if stopped_daemon.get("status") != "stopped":
                    raise DatabaseMaintenanceError("APRIL daemon could not be stopped safely.")
            stopped = manager.stop()
            running = stopped.runtime.running or stopped.api.running
            if running:
                raise DatabaseMaintenanceError("APRIL services could not be stopped safely.")
        result = restore_backup(
            settings.database_path,
            input_path,
            home=settings.home,
            services_running=running,
            audit=audit_logger_for_settings(settings),
        )
    except (ConfigError, DatabaseMaintenanceError, OSError) as exc:
        console.print(
            f"[red]Database restore failed ({type(exc).__name__}); "
            "the active database was not left partially restored.[/red]"
        )
        raise typer.Exit(1) from exc
    console.print(
        f"Database restored and reopened successfully; schema={result.schema_version}. "
        f"Rollback backup retained as {result.rollback_backup.name}."
    )
    if stop_services:
        console.print("APRIL services remain stopped; start them after reviewing the result.")
