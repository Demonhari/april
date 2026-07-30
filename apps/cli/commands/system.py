"""Health, mute, session, and feedback CLI commands."""

from __future__ import annotations

import asyncio
from typing import Any

import typer

from apps.cli.client import ApiOfflineError, ApiResponseError
from apps.cli.groups import app
from apps.cli.render import console, print_jsonish
from april_common.settings import get_settings


def client() -> Any:
    from apps.cli import main as cli_main

    return cli_main.client()


def run(coro: Any) -> Any:
    from apps.cli import main as cli_main

    return cli_main.run(coro)


@app.command()
def health() -> None:
    data = run(client().get("/health", auth=False))
    print_jsonish(data)


@app.command()
def mute(
    off: bool = typer.Option(False, "--off", help="Release the hard mute."),
) -> None:
    """Hard-mute the Sentinel: the microphone stream is fully released.

    The change goes through the Core API so it is audited. Only when the API
    is unreachable does the CLI flip the local flag directly, and it says so
    explicitly (unaudited_fallback=true) instead of silently falling back.
    """
    muted = not off
    try:
        data = asyncio.run(client().post("/wake/mute", {"muted": muted}))
    except ApiResponseError as exc:
        # A reachable API refusal must never be bypassed by a local write.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except ApiOfflineError:
        from services.wake.sentinel import MuteSwitch

        switch = MuteSwitch(get_settings().mute_flag_path)
        if muted:
            switch.mute()
        else:
            switch.unmute()
        console.print(
            "[yellow]APRIL API is offline; the mute flag was changed locally "
            "WITHOUT an audit trail (unaudited_fallback=true).[/yellow]"
        )
        print_jsonish({"muted": switch.is_muted(), "audited": False, "unaudited_fallback": True})
    else:
        print_jsonish(data)
    if muted:
        console.print("Voice hard-muted. The Sentinel releases the microphone.")
    else:
        console.print("Voice unmuted. The Sentinel may reopen the microphone.")


@app.command()
def sessions() -> None:
    """List recent wake/conversation sessions."""
    data = run(client().get("/sessions"))
    print_jsonish(data)


@app.command()
def good() -> None:
    """Mark the last answer in the active session as good."""
    data = run(client().post("/feedback", {"rating": "good"}))
    print_jsonish(data)


@app.command()
def bad(
    reason: str = typer.Argument(None, help="Optional short reason."),
) -> None:
    """Mark the last answer in the active session as bad."""
    data = run(client().post("/feedback", {"rating": "bad", "reason": reason}))
    print_jsonish(data)
