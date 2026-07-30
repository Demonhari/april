from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import typer

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.evals import run_fake_brain_eval, run_real_brain_eval
from apps.runner.wake_live import run_sentinel_live_verification

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


@_registry.conversation_app.command("delete")
def conversation_delete(
    ctx: typer.Context,
    conversation_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["conversation", "delete", conversation_id],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.reminder_app.command("list")
def reminder_list(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["reminder", "list"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.reminder_app.command("create")
def reminder_create(
    ctx: typer.Context,
    content: str,
    due_at: str | None = typer.Option(None, "--due-at"),
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    args = ["reminder", "create", content]
    if due_at:
        args.extend(["--due-at", due_at])
    _composition_api._delegate(
        args,
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.reminder_app.command("delete")
def reminder_delete(
    ctx: typer.Context,
    reminder_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["reminder", "delete", reminder_id],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.task_app.command("list")
def task_list(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["task", "list"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.eval_app.command("brain")
def eval_brain(
    fake: bool = typer.Option(False, "--fake", help="Run deterministic fake Brain eval."),
    real_model: Path | None = typer.Option(None, "--real-model"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if fake:
        results = run_fake_brain_eval(_composition_api._manager().home)
    elif real_model is not None:
        if not real_model.expanduser().exists():
            console.print(f"[red]GGUF path does not exist: {real_model}[/red]")
            raise typer.Exit(1)
        results = run_real_brain_eval(_composition_api._manager().home, real_model)
    else:
        console.print("[red]Use --fake or --real-model /path/to/model.gguf.[/red]")
        raise typer.Exit(1)
    if json_output:
        console.print_json(data={"results": [result.model_dump() for result in results]})
    else:
        _composition_api._print_brain_eval(results)
    if not all(result.ok for result in results):
        raise typer.Exit(1)
