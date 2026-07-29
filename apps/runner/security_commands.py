from __future__ import annotations

import asyncio
import platform
from pathlib import Path

import typer

from apps.cli.render import console
from apps.daemon.apriald import (
    read_daemon_status,
    start_daemon_background,
    stop_daemon,
)
from apps.runner.service_manager import AprilServiceManager
from april_common.audit import AuditLogger, CredentialAuditAnchor
from april_common.credentials import (
    CredentialStore,
    CredentialStoreError,
    select_credential_store,
)
from april_common.errors import ConfigError
from april_common.settings import AprilSettings, load_settings, reset_settings_cache
from april_common.token_setup import migrate_legacy_credentials, rotate_credentials
from services.memory.database import Database
from services.memory.encryption import (
    MemoryEncryptionError,
    provision_memory_key,
    rotate_memory_key,
)
from services.memory.migrations import run_migrations

security_app = typer.Typer(help="Local credential and security operations.")
credentials_app = typer.Typer(help="Migrate and rotate APRIL credentials.")
memory_encryption_app = typer.Typer(help="Manage optional sensitive-memory encryption.")
security_app.add_typer(credentials_app, name="credentials")
security_app.add_typer(memory_encryption_app, name="memory-encryption")


def _store_for_command(
    settings: AprilSettings,
    *,
    backend: str | None,
    file_path: Path | None,
) -> CredentialStore:
    selected = backend or settings.security.credential_store
    if selected == "auto":
        selected = "keychain" if platform.system() == "Darwin" else "memory"
    if selected == "memory" and settings.environment != "test":
        raise ConfigError(
            "Non-macOS development requires an explicit --store file and "
            "--credential-file path outside the repository."
        )
    configured_file = file_path or settings.security.credential_file_path
    return select_credential_store(
        backend=selected,
        environment=settings.environment,
        repository_root=settings.home,
        file_path=(
            configured_file.expanduser().resolve(strict=False)
            if configured_file is not None
            else None
        ),
    )


def _base_settings(*, legacy_migration: bool = False) -> AprilSettings:
    """Load legacy settings without forcing a not-yet-configured secure backend."""
    return load_settings(legacy_credential_migration=legacy_migration)


@credentials_app.command("migrate")
def credentials_migrate(
    backend: str | None = typer.Option(None, "--store"),
    credential_file: Path | None = typer.Option(None, "--credential-file"),
) -> None:
    try:
        settings = _base_settings(legacy_migration=True)
        store = _store_for_command(
            settings,
            backend=backend,
            file_path=credential_file,
        )
        result = migrate_legacy_credentials(
            home=settings.home,
            store=store,
            legacy_audit_anchor_file=settings.audit_path.with_name(
                f"{settings.audit_path.name}.anchor"
            ),
        )
    except (ConfigError, CredentialStoreError, OSError) as exc:
        console.print(
            f"[red]Credential migration failed ({type(exc).__name__}); "
            "legacy files were left unchanged.[/red]"
        )
        raise typer.Exit(1) from exc
    reset_settings_cache()
    console.print(
        f"Credential migration: {result.status}; store={result.store}; "
        f"migrated={', '.join(result.migrated) if result.migrated else 'none'}."
    )
    console.print("Credential values were not displayed.")


@credentials_app.command("rotate")
def credentials_rotate(
    api: bool = typer.Option(False, "--api"),
    runtime: bool = typer.Option(False, "--runtime"),
    all_credentials: bool = typer.Option(False, "--all"),
    restart_services: bool = typer.Option(False, "--restart-services"),
    backend: str | None = typer.Option(None, "--store"),
    credential_file: Path | None = typer.Option(None, "--credential-file"),
) -> None:
    if all_credentials and (api or runtime):
        console.print("[red]Use --all by itself, or select --api/--runtime.[/red]")
        raise typer.Exit(2)
    rotate_api = all_credentials or api
    rotate_runtime = all_credentials or runtime
    try:
        settings = _base_settings()
        store = _store_for_command(
            settings,
            backend=backend,
            file_path=credential_file,
        )
        audit = AuditLogger(
            settings.audit_path,
            anchor=CredentialAuditAnchor(store),
        )

        def restart_after_rotation() -> None:
            if not restart_services:
                return
            reset_settings_cache()
            daemon_status = read_daemon_status(settings)
            if isinstance(daemon_status.get("pid"), int):
                stopped = stop_daemon(settings)
                if stopped.get("status") != "stopped":
                    raise CredentialStoreError(
                        "apriald could not be stopped for credential rotation."
                    )
                refreshed = load_settings(root=settings.home)
                start_daemon_background(refreshed)
                return
            manager = AprilServiceManager(home=settings.home)
            status = manager.restart(fake_backend=settings.runtime.backend == "fake")
            if not status.ok:
                raise CredentialStoreError(
                    "Services did not become healthy after credential rotation."
                )

        result = rotate_credentials(
            store=store,
            rotate_api=rotate_api,
            rotate_runtime=rotate_runtime,
            audit=audit,
            commit_callback=restart_after_rotation,
        )
        reset_settings_cache()
    except (ConfigError, CredentialStoreError, OSError, RuntimeError) as exc:
        console.print(
            f"[red]Credential rotation failed ({type(exc).__name__}); "
            "no credential values were displayed.[/red]"
        )
        raise typer.Exit(1) from exc
    console.print(f"Rotated: {', '.join(result.rotated)} in {result.store}.")
    if restart_services:
        console.print("Required services were restarted and became healthy.")
    else:
        console.print(f"Restart required: {', '.join(result.restart_services)}.")
    console.print("Credential values were not displayed.")


@memory_encryption_app.command("provision")
def memory_encryption_provision(
    backend: str | None = typer.Option(None, "--store"),
    credential_file: Path | None = typer.Option(None, "--credential-file"),
) -> None:
    """Provision an authenticated-encryption key without enabling the feature."""
    try:
        settings = _base_settings()
        store = _store_for_command(
            settings,
            backend=backend,
            file_path=credential_file,
        )
        provision_memory_key(store)
    except (ConfigError, CredentialStoreError, MemoryEncryptionError, OSError) as exc:
        console.print(
            f"[red]Memory encryption key provisioning failed ({type(exc).__name__}).[/red]"
        )
        raise typer.Exit(1) from exc
    console.print(f"Memory encryption key is available in {store.backend_name}.")
    console.print(
        "No memory was changed. Set memory.sensitive_encryption_enabled=true "
        "after reviewing the local configuration."
    )


@memory_encryption_app.command("rotate")
def memory_encryption_rotate(
    backend: str | None = typer.Option(None, "--store"),
    credential_file: Path | None = typer.Option(None, "--credential-file"),
) -> None:
    """Re-encrypt protected memory rows under a new staged key."""

    async def operation() -> dict[str, int | str]:
        settings = _base_settings()
        if not settings.memory.sensitive_encryption_enabled:
            raise ConfigError("Sensitive-memory encryption is disabled.")
        store = _store_for_command(
            settings,
            backend=backend,
            file_path=credential_file,
        )
        database = Database(settings.database_path)
        await database.connect()
        try:
            await run_migrations(database)
            result = await rotate_memory_key(settings, database, store=store)
        finally:
            await database.close()
        audit = AuditLogger(settings.audit_path, anchor=CredentialAuditAnchor(store))
        audit.write(
            {
                "event_type": "sensitive_memory_key_rotated",
                "actor": "local-operator",
                "rotated_records": result["rotated_records"],
            }
        )
        return result

    try:
        result = asyncio.run(operation())
    except (ConfigError, CredentialStoreError, MemoryEncryptionError, OSError) as exc:
        console.print(
            f"[red]Memory encryption key rotation failed ({type(exc).__name__}); "
            "encrypted rows were not discarded.[/red]"
        )
        raise typer.Exit(1) from exc
    console.print(f"Rotated {result['rotated_records']} encrypted memory row(s).")
    console.print("No key or plaintext value was displayed.")
