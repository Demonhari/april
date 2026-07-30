from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import typer

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.wake_live import run_sentinel_live_verification
from april_common.effective_config import load_agents_file, load_permissions_file, load_tools_file
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelRegistry
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.user_profile import UserProfileStore

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


@_registry.agent_app.command("run")
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
    _composition_api._delegate(
        args,
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.config_app.command("validate")
def config_validate() -> None:
    errors = _composition_api.validate_configuration(_composition_api._manager().home)
    if errors:
        console.print("[red]APRIL configuration is invalid.[/red]")
        for error in errors:
            console.print(f"- {error}")
        raise typer.Exit(1)
    console.print("[green]APRIL configuration is valid.[/green]")


@_registry.config_app.command("inspect")
def config_inspect() -> None:
    errors = _composition_api.validate_configuration(_composition_api._manager().home)
    if errors:
        console.print("[red]APRIL configuration is invalid.[/red]")
        for error in errors:
            console.print(f"- {error}")
        raise typer.Exit(1)
    settings = load_settings(root=_composition_api._manager().home)
    home = _composition_api._manager().home
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
        settings = load_settings(root=_composition_api._manager().home)

        async with Database(settings.database_path) as database:
            await run_migrations(database)
            return await operation(UserProfileStore(database))

    return asyncio.run(_run())


@_registry.user_profile_app.command("show")
def profile_show() -> None:
    """Inspect the local user profile. It is stored only on this machine."""
    profile = _run_profile_op(lambda store: store.get())
    if profile is None:
        console.print("No local profile is set. Use `run april profile set --display-name ...`.")
        return
    console.print_json(data=profile.model_dump())


@_registry.user_profile_app.command("set")
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


@_registry.user_profile_app.command("delete")
def profile_delete() -> None:
    """Delete the local user profile."""
    deleted = _run_profile_op(lambda store: store.delete())
    console.print(f"Deleted local profile: {deleted}")
