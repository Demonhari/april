from __future__ import annotations

from typing import Any

import typer

from apps.cli.groups import daemon_app
from apps.cli.render import print_jsonish
from april_common.settings import get_settings


def client() -> Any:
    from apps.cli import main as cli_main

    return cli_main.client()


def run(coro: Any) -> Any:
    from apps.cli import main as cli_main

    return cli_main.run(coro)


@daemon_app.command("install")
def daemon_install() -> None:
    from apps.daemon.launchd import LaunchdManager

    manager = LaunchdManager(get_settings())
    path = manager.install()
    print_jsonish({"installed": True, "plist_path": str(path), "load": manager.bootstrap()})


@daemon_app.command("uninstall")
def daemon_uninstall() -> None:
    from apps.daemon.launchd import LaunchdManager

    manager = LaunchdManager(get_settings())
    unload = manager.bootout()
    removed = manager.uninstall()
    print_jsonish({"removed": removed, "unload": unload})


@daemon_app.command("start")
def daemon_start() -> None:
    from apps.daemon.apriald import start_daemon_background, wait_for_core_health
    from apps.daemon.launchd import LaunchdManager
    from april_common.audit import AuditStartupBlocked

    settings = get_settings()
    manager = LaunchdManager(settings)
    launchd = manager.status()
    if launchd.get("supported") is True and launchd.get("installed") is True:
        action = manager.kickstart() if launchd.get("loaded") is True else manager.bootstrap()
        if action.get("loaded") is True or action.get("started") is True:
            health = wait_for_core_health(settings)
            print_jsonish({"status": "running", "launchd": action, "health": health})
            return
        print_jsonish({"status": "degraded", "launchd": action})
        raise typer.Exit(1)
    try:
        print_jsonish(start_daemon_background(settings))
    except AuditStartupBlocked as exc:
        print_jsonish(
            {
                "blocker": "audit_chain_integrity",
                **exc.decision.to_dict(),
                "status": "blocked",
            }
        )
        raise typer.Exit(1) from exc


@daemon_app.command("stop")
def daemon_stop() -> None:
    from apps.daemon.apriald import stop_daemon
    from apps.daemon.launchd import LaunchdManager

    settings = get_settings()
    manager = LaunchdManager(settings)
    launchd = manager.status()
    if launchd.get("supported") is True and launchd.get("loaded") is True:
        result = manager.bootout()
        print_jsonish(
            {"status": "stopped" if result.get("changed") else "degraded", "launchd": result}
        )
        if not result.get("changed"):
            raise typer.Exit(1)
        return
    print_jsonish(stop_daemon(settings))


@daemon_app.command("status")
def daemon_status() -> None:
    from apps.daemon.apriald import read_daemon_status
    from apps.daemon.launchd import LaunchdManager
    from april_common.audit import audit_startup_decision

    settings = get_settings()
    status = read_daemon_status(settings)
    decision = audit_startup_decision(settings)
    if not decision.accepted:
        status.update(
            {
                "blocker": "audit_chain_integrity",
                **decision.to_dict(),
                "status": "blocked",
            }
        )
    status["launchd"] = LaunchdManager(settings).status()
    print_jsonish(status)
