from __future__ import annotations

from typing import TypeVar

import typer

from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.wake_live import run_sentinel_live_verification

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


@_registry.jobs_app.command(
    "submit",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def jobs_submit(ctx: typer.Context, job_type: str) -> None:
    _composition_api._delegate(["jobs", "submit", job_type, *ctx.args], fake=False)


@_registry.jobs_app.command(
    "list",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def jobs_list(ctx: typer.Context) -> None:
    _composition_api._delegate(["jobs", "list", *ctx.args], fake=False)


@_registry.jobs_app.command(
    "show",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def jobs_show(ctx: typer.Context, job_id: str) -> None:
    _composition_api._delegate(["jobs", "show", job_id, *ctx.args], fake=False)


@_registry.jobs_app.command(
    "cancel",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def jobs_cancel(ctx: typer.Context, job_id: str) -> None:
    _composition_api._delegate(["jobs", "cancel", job_id, *ctx.args], fake=False)


@_registry.jobs_app.command(
    "retry",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def jobs_retry(ctx: typer.Context, job_id: str) -> None:
    _composition_api._delegate(["jobs", "retry", job_id, *ctx.args], fake=False)
