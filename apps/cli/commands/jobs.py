"""Durable job CLI commands."""

from __future__ import annotations

import json
import time
from typing import Any

import typer

from apps.cli.groups import jobs_app
from apps.cli.render import console, print_jsonish


def client() -> Any:
    from apps.cli import main as cli_main

    return cli_main.client()


def run(coro: Any) -> Any:
    from apps.cli import main as cli_main

    return cli_main.run(coro)


def _wait_for_job(job_id: str, *, timeout_seconds: float) -> dict[str, Any]:
    terminal = {"cancelled", "succeeded", "failed", "interrupted"}
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            job = run(client().get(f"/jobs/{job_id}"))
            if str(job.get("status")) in terminal:
                return job
            if time.monotonic() >= deadline:
                raise typer.BadParameter("Timed out waiting; the durable job is still running.")
            time.sleep(0.25)
    except KeyboardInterrupt:
        console.print(
            "[yellow]Stopped waiting. The durable job was not cancelled; "
            f"use `run april jobs cancel {job_id}` to request cancellation.[/yellow]"
        )
        raise typer.Exit(130) from None


@jobs_app.command("submit")
def jobs_submit(
    job_type: str,
    payload: str = typer.Option("{}", "--payload", help="Bounded JSON object payload."),
    project_id: str | None = typer.Option(None, "--project-id"),
    conversation_id: str | None = typer.Option(None, "--conversation-id"),
    approval_id: str | None = typer.Option(None, "--approval-id"),
    wait: bool = typer.Option(False, "--wait"),
    json_output: bool = typer.Option(False, "--json"),
    wait_timeout: float = typer.Option(3600.0, "--wait-timeout", min=1.0, max=86400.0),
) -> None:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("--payload must be valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise typer.BadParameter("--payload must decode to an object.")
    request: dict[str, Any] = {"job_type": job_type, "payload": decoded}
    if project_id is not None:
        request["project_id"] = project_id
    if conversation_id is not None:
        request["conversation_id"] = conversation_id
    if approval_id is not None:
        request["approval_id"] = approval_id
    job = run(client().post("/jobs", request))
    if wait:
        job = _wait_for_job(str(job["id"]), timeout_seconds=wait_timeout)
    if json_output:
        console.print_json(data=job)
    else:
        console.print(f"{job['id']} {job['job_type']} {job['status']}")


@jobs_app.command("list")
def jobs_list(
    project_id: str | None = typer.Option(None, "--project-id"),
    limit: int = typer.Option(25, "--limit", min=1, max=100),
    offset: int = typer.Option(0, "--offset", min=0, max=10000),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if project_id is not None:
        params["project_id"] = project_id
    data = run(client().get("/jobs", params=params))
    if json_output:
        console.print_json(data=data)
        return
    for job in data["jobs"]:
        console.print(f"{job['id']} {job['job_type']} {job['status']} {job['progress_percent']}%")


@jobs_app.command("show")
def jobs_show(job_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    job = run(client().get(f"/jobs/{job_id}"))
    if json_output:
        console.print_json(data=job)
    else:
        print_jsonish(job)


@jobs_app.command("cancel")
def jobs_cancel(job_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    data = run(client().post(f"/jobs/{job_id}/cancel", {}))
    if json_output:
        console.print_json(data=data)
    else:
        console.print(f"{data['job']['id']} {data['job']['status']}")


@jobs_app.command("retry")
def jobs_retry(job_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    data = run(client().post(f"/jobs/{job_id}/retry", {}))
    if json_output:
        console.print_json(data=data)
    else:
        console.print(f"{data['job']['id']} {data['job']['status']}")
