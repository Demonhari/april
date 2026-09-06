from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from fastapi import FastAPI
from typer.testing import CliRunner

from apps.cli.main import app as cli_app
from apps.daemon.apriald import (
    AprialdSupervisor,
    AuditStartupBlocked,
    autostart_if_needed,
    daemon_status_path,
    read_daemon_status,
    start_daemon_background,
)
from apps.daemon.launchd import LaunchdManager
from apps.runner.main import app as runner_app
from apps.runner.service_manager import AprilServiceManager
from april_common.audit import audit_startup_decision


def _corrupt_audit(settings) -> None:
    settings.audit_path.parent.mkdir(parents=True, exist_ok=True)
    settings.audit_path.write_bytes(b"{}\n{}\n{}\n")
    settings.audit_path.with_name(f"{settings.audit_path.name}.anchor").unlink(missing_ok=True)


def _unavailable_audit(settings) -> None:
    settings.audit_path.parent.mkdir(parents=True, exist_ok=True)
    if settings.audit_path.exists() and not settings.audit_path.is_dir():
        settings.audit_path.unlink()
    settings.audit_path.mkdir(exist_ok=True)


def _forbidden_popen(calls: list[object]):
    def forbidden(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("blocked startup must not spawn a child")

    return forbidden


def test_shared_startup_decision_distinguishes_corrupt_unavailable_and_empty(settings_tmp) -> None:
    decision = audit_startup_decision(settings_tmp)
    assert decision.accepted is True
    assert decision.status == "valid"

    _corrupt_audit(settings_tmp)
    corrupt = audit_startup_decision(settings_tmp)
    assert corrupt.accepted is False
    assert corrupt.status == "corrupt"
    assert "invalid_schema" in corrupt.issue_codes
    assert "recover" in " ".join(corrupt.next_commands)

    _unavailable_audit(settings_tmp)
    unavailable = audit_startup_decision(settings_tmp)
    assert unavailable.accepted is False
    assert unavailable.status == "unavailable"
    assert "recover" not in " ".join(unavailable.next_commands)


def test_service_manager_blocks_before_processes_pids_or_lifecycle(settings_tmp) -> None:
    _corrupt_audit(settings_tmp)
    popen_calls: list[object] = []
    manager = AprilServiceManager(
        home=settings_tmp.home,
        popen_factory=_forbidden_popen(popen_calls),
        health_getter=lambda _url, _timeout: (_ for _ in ()).throw(
            AssertionError("blocked startup must not poll health")
        ),
    )

    with pytest.raises(AuditStartupBlocked) as error:
        manager.start(fake_backend=True)

    assert error.value.decision.status == "corrupt"
    assert popen_calls == []
    assert not manager.runtime_pid_path.exists()
    assert not manager.api_pid_path.exists()
    assert not manager.lifecycle_path.exists()


def test_restart_checks_audit_before_stopping_existing_services(settings_tmp, monkeypatch) -> None:
    _corrupt_audit(settings_tmp)
    manager = AprilServiceManager(home=settings_tmp.home)
    stopped = False

    def forbidden_stop() -> object:
        nonlocal stopped
        stopped = True
        raise AssertionError("blocked restart must not stop existing services")

    monkeypatch.setattr(manager, "stop", forbidden_stop)
    with pytest.raises(AuditStartupBlocked):
        manager.restart(fake_backend=True)
    assert stopped is False


@pytest.mark.parametrize(
    "args",
    [
        ["april"],
        ["april", "chat"],
        ["april", "chat", "--fake"],
        ["april", "ask", "hello"],
        ["april", "desktop", "--no-open"],
        ["april", "start"],
        ["april", "start", "--fake"],
        ["april", "start", "--preflight", "--json"],
    ],
)
def test_runner_operational_paths_block_corrupt_audit_before_spawn(
    settings_tmp, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    _corrupt_audit(settings_tmp)
    popen_calls: list[object] = []
    manager = AprilServiceManager(
        home=settings_tmp.home,
        popen_factory=_forbidden_popen(popen_calls),
        health_getter=lambda _url, _timeout: (_ for _ in ()).throw(
            AssertionError("blocked startup must not poll health")
        ),
    )
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)

    result = CliRunner().invoke(runner_app, args)

    assert result.exit_code == 1, result.output
    assert "audit chain is corrupt" in result.output or "corrupt" in result.output
    assert "run april audit verify --json" in result.output
    assert "run april audit recover" in result.output
    assert "did not become healthy" not in result.output
    assert "15.0" not in result.output
    assert popen_calls == []


def test_bare_cli_and_voice_autostart_block_without_generic_timeout(
    settings_tmp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corrupt_audit(settings_tmp)
    monkeypatch.setattr("apps.cli.main.get_settings", lambda: settings_tmp)
    popen_calls: list[object] = []
    monkeypatch.setattr("apps.daemon.apriald.subprocess.Popen", _forbidden_popen(popen_calls))

    bare = CliRunner().invoke(cli_app, [])
    voice = CliRunner().invoke(cli_app, ["voice", "listen"])

    for result in (bare, voice):
        assert result.exit_code == 1, result.output
        assert "audit chain is corrupt" in result.output
        assert "No operational services were started" in result.output
        assert "did not become healthy" not in result.output
        assert "15.0" not in result.output
    assert popen_calls == []


def test_one_shot_cli_message_uses_the_same_blocked_autostart_gate(
    settings_tmp, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _corrupt_audit(settings_tmp)
    import apps.cli.main as cli_main

    monkeypatch.setattr(cli_main, "get_settings", lambda: settings_tmp)
    monkeypatch.setattr("apps.daemon.apriald.subprocess.Popen", _forbidden_popen([]))
    monkeypatch.setattr(sys, "argv", ["april", "some question"])

    with pytest.raises(typer.Exit) as error:
        cli_main.main()

    assert error.value.exit_code == 1
    output = capsys.readouterr().out
    assert "audit chain is corrupt" in output
    assert "run april audit recover" in output


def test_direct_daemon_start_and_autostart_block_corrupt_and_unavailable(settings_tmp, monkeypatch):
    for prepare, expected_status, recovery_allowed in (
        (_corrupt_audit, "corrupt", True),
        (_unavailable_audit, "unavailable", False),
    ):
        prepare(settings_tmp)
        popen_calls: list[object] = []
        monkeypatch.setattr("apps.daemon.apriald.subprocess.Popen", _forbidden_popen(popen_calls))
        for starter in (start_daemon_background, autostart_if_needed):
            with pytest.raises(AuditStartupBlocked) as error:
                starter(settings_tmp, fake_backend=True)  # type: ignore[call-arg]
            assert error.value.decision.status == expected_status
        assert popen_calls == []
        status = read_daemon_status(settings_tmp)
        assert status["status"] == "blocked"
        assert status["blocker"] == "audit_chain_integrity"
        assert status["audit_status"] == expected_status
        assert recovery_allowed == ("recover" in " ".join(status["next_commands"]))
        assert not (settings_tmp.home / "data" / "apriald.pid").exists()


def test_daemon_start_and_status_cli_report_audit_block_without_launchd_or_children(
    settings_tmp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corrupt_audit(settings_tmp)

    class NoLaunchd:
        def __init__(self, _settings) -> None:
            pass

        def status(self) -> dict[str, object]:
            return {"supported": False, "loaded": False}

    popen_calls: list[object] = []
    monkeypatch.setattr("apps.cli.commands.daemon.get_settings", lambda: settings_tmp)
    monkeypatch.setattr("apps.daemon.launchd.LaunchdManager", NoLaunchd)
    monkeypatch.setattr("apps.daemon.apriald.subprocess.Popen", _forbidden_popen(popen_calls))

    started = CliRunner().invoke(cli_app, ["daemon", "start"])
    status = CliRunner().invoke(cli_app, ["daemon", "status"])

    assert started.exit_code == 1, started.output
    assert "blocked" in started.output
    assert "corrupt" in started.output
    assert status.exit_code == 0, status.output
    assert "audit_chain_integrity" in status.output
    assert "corrupt" in status.output
    assert popen_calls == []


@pytest.mark.asyncio
async def test_supervisor_writes_safe_blocked_status_without_children(settings_tmp) -> None:
    _corrupt_audit(settings_tmp)
    started: list[str] = []

    async def process_factory(spec) -> object:
        started.append(spec.name)
        raise AssertionError("blocked supervisor must not spawn children")

    supervisor = AprialdSupervisor(
        settings_tmp,
        process_factory=process_factory,
        health_checker=lambda _spec: None,  # type: ignore[arg-type]
        sleep=lambda _seconds: _immediate_sleep(),
    )
    with pytest.raises(AuditStartupBlocked):
        await supervisor.start()

    status_path = daemon_status_path(settings_tmp)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["blocker"] == "audit_chain_integrity"
    assert payload["audit_status"] == "corrupt"
    assert "private" not in json.dumps(payload).lower()
    assert started == []
    assert not (settings_tmp.home / "data" / "apriald.pid").exists()


async def _immediate_sleep() -> None:
    return None


def test_launchd_bootstrap_and_kickstart_refuse_without_launchctl(settings_tmp, tmp_path: Path):
    _corrupt_audit(settings_tmp)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(list(argv))
        return SimpleNamespace(returncode=0)

    manager = LaunchdManager(
        settings_tmp,
        user_home=tmp_path / "user",
        runner=runner,
        platform="darwin",
        uid=501,
    )

    bootstrap = manager.bootstrap()
    kickstart = manager.kickstart()
    assert bootstrap["status"] == "blocked"
    assert kickstart["status"] == "blocked"
    assert bootstrap["audit_status"] == "corrupt"
    assert calls == []


def test_direct_api_main_refuses_before_uvicorn_but_app_construction_remains_available(
    settings_tmp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corrupt_audit(settings_tmp)
    import services.api.server as api_server

    uvicorn_calls: list[object] = []
    monkeypatch.setattr(api_server, "get_settings", lambda: settings_tmp)
    monkeypatch.setattr(
        api_server.uvicorn,
        "run",
        lambda *args, **kwargs: uvicorn_calls.append(args),
    )

    assert isinstance(api_server.create_app(), FastAPI)
    with pytest.raises(SystemExit) as error:
        api_server.main()
    assert error.value.code == 1
    assert uvicorn_calls == []
