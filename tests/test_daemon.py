from __future__ import annotations

import json
import os
import plistlib
import signal
import subprocess
from pathlib import Path

import pytest

from apps.daemon.apriald import (
    AprialdSupervisor,
    DaemonLock,
    daemon_pid_path,
    daemon_status_path,
    read_daemon_status,
    stop_daemon,
    wait_for_core_health,
)
from apps.daemon.launchd import LaunchdManager
from services.pool.governor import ResourceGovernor, ResourcePolicy, ResourceSignals


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class FakeFactory:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.processes: list[FakeProcess] = []

    async def __call__(self, spec) -> FakeProcess:
        process = FakeProcess(pid=10_000 + len(self.processes))
        self.started.append(spec.name)
        self.processes.append(process)
        return process


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FixedSignals:
    def __init__(self, signals: ResourceSignals) -> None:
        self.signals = signals

    def sample(self) -> ResourceSignals:
        return self.signals


async def _healthy(_spec) -> bool:
    return True


async def _unhealthy_api(spec) -> bool:
    return spec.name != "api"


async def _no_sleep(_seconds: float) -> None:
    return None


def test_daemon_lock_is_single_instance(tmp_path: Path) -> None:
    path = tmp_path / "data" / "apriald.lock"
    first = DaemonLock(path)
    second = DaemonLock(path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            second.acquire()
    finally:
        first.release()


def _permissive_governor(settings) -> ResourceGovernor:
    return ResourceGovernor(
        settings,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=16.0,
                cpu_load_percent=5.0,
                on_ac_power=True,
                user_idle_seconds=600.0,
            )
        ),
        policy=ResourcePolicy(min_ram_headroom_gb=1.0, max_cpu_load_percent=90.0),
    )


def _denying_governor(settings) -> ResourceGovernor:
    return ResourceGovernor(
        settings,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=0.2,
                cpu_load_percent=96.0,
                on_ac_power=True,
                user_idle_seconds=600.0,
            )
        ),
        policy=ResourcePolicy(min_ram_headroom_gb=1.0, max_cpu_load_percent=80.0),
    )


def _supervisor(settings, factory: FakeFactory, clock: ManualClock, **kwargs) -> AprialdSupervisor:
    return AprialdSupervisor(
        settings,
        process_factory=factory,
        health_checker=_healthy,
        governor=_permissive_governor(settings),
        sleep=_no_sleep,
        clock=clock,
        **kwargs,
    )


def _voice_wake_enabled(settings):
    return settings.model_copy(
        update={
            "voice": settings.voice.model_copy(update={"enabled": True}),
            "wake": settings.wake.model_copy(update={"enabled": True}),
        }
    )


@pytest.mark.asyncio
async def test_supervisor_default_safe_config_excludes_sentinel(settings_tmp) -> None:
    factory = FakeFactory()
    supervisor = _supervisor(settings_tmp, factory, ManualClock())
    await supervisor.start()
    try:
        assert factory.started == ["runtime", "api"]
        assert set(supervisor.children) == {"runtime", "api"}
        health = await supervisor.health()
        assert health.status == "ok"
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_daemon_status_reports_running_child_details(settings_tmp) -> None:
    factory = FakeFactory()
    supervisor = _supervisor(settings_tmp, factory, ManualClock())
    await supervisor.start()
    try:
        status = read_daemon_status(settings_tmp)
        assert status["status"] == "running"
        assert status["pid"] == os.getpid()
        assert status["details_available"] is True
        assert status["governor"] == {"allowed": True, "reasons": []}
        children = {child["name"]: child for child in status["children"]}  # type: ignore[index]
        assert set(children) == {"runtime", "api"}
        assert children["runtime"]["status"] == "running"
        assert children["runtime"]["pid"] == 10_000
        assert children["runtime"]["restarts"] == 0
        assert children["runtime"]["last_exit_code"] is None
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_daemon_status_reports_degraded_child(settings_tmp) -> None:
    factory = FakeFactory()
    supervisor = AprialdSupervisor(
        settings_tmp,
        process_factory=factory,
        health_checker=_unhealthy_api,
        governor=_permissive_governor(settings_tmp),
        sleep=_no_sleep,
        clock=ManualClock(),
    )
    await supervisor.start()
    try:
        status = read_daemon_status(settings_tmp)
        assert status["status"] == "degraded"
        children = {child["name"]: child for child in status["children"]}  # type: ignore[index]
        assert children["api"]["status"] == "degraded"
        assert children["api"]["degraded_reason"] == "health_check_failed"
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_daemon_status_reports_paused_governor_state(settings_tmp) -> None:
    factory = FakeFactory()
    supervisor = AprialdSupervisor(
        settings_tmp,
        process_factory=factory,
        health_checker=_healthy,
        governor=_denying_governor(settings_tmp),
        sleep=_no_sleep,
        clock=ManualClock(),
    )
    await supervisor.start()
    try:
        status = read_daemon_status(settings_tmp)
        assert status["status"] == "paused"
        assert status["governor"] == {
            "allowed": False,
            "reasons": ["ram_headroom_below_policy", "cpu_load_above_policy"],
        }
        assert factory.started == []
        children = {child["name"]: child for child in status["children"]}  # type: ignore[index]
        assert {child["status"] for child in children.values()} == {"paused"}
        assert children["runtime"]["paused_reason"] == (
            "ram_headroom_below_policy,cpu_load_above_policy"
        )
    finally:
        await supervisor.stop()


def test_daemon_status_reports_stale_with_last_details(settings_tmp) -> None:
    daemon_pid_path(settings_tmp).parent.mkdir(parents=True, exist_ok=True)
    daemon_pid_path(settings_tmp).write_text("999999\n", encoding="utf-8")
    daemon_status_path(settings_tmp).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "degraded",
                "pid": 999999,
                "generated_at": "2026-07-04T00:00:00Z",
                "children": [
                    {
                        "name": "runtime",
                        "status": "down",
                        "pid": 123,
                        "restarts": 2,
                        "last_exit_code": 1,
                        "detail": "restart_scheduled_in=2.0s failures=1",
                        "paused_reason": None,
                        "degraded_reason": None,
                    }
                ],
                "governor": {"allowed": True, "reasons": []},
            }
        ),
        encoding="utf-8",
    )

    status = read_daemon_status(settings_tmp)

    assert status["status"] == "stale"
    assert status["pid"] == 999999
    assert status["details_available"] is True
    assert status["children"][0]["name"] == "runtime"  # type: ignore[index]
    assert status["supervisor_status"] == "degraded"


@pytest.mark.asyncio
async def test_supervisor_includes_sentinel_when_voice_and_wake_enabled(settings_tmp) -> None:
    factory = FakeFactory()
    supervisor = _supervisor(_voice_wake_enabled(settings_tmp), factory, ManualClock())
    await supervisor.start()
    try:
        assert factory.started == ["runtime", "api", "sentinel"]
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_supervisor_excludes_sentinel_when_only_wake_enabled(settings_tmp) -> None:
    wake_only = settings_tmp.model_copy(
        update={"wake": settings_tmp.wake.model_copy(update={"enabled": True})}
    )
    factory = FakeFactory()
    supervisor = _supervisor(wake_only, factory, ManualClock())
    await supervisor.start()
    try:
        assert factory.started == ["runtime", "api"]
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_supervisor_starts_children_degrades_and_restarts_with_backoff(settings_tmp) -> None:
    factory = FakeFactory()
    clock = ManualClock()
    supervisor = _supervisor(_voice_wake_enabled(settings_tmp), factory, clock)
    await supervisor.start()
    try:
        assert factory.started == ["runtime", "api", "sentinel"]
        health = await supervisor.health()
        assert health.status == "ok"

        runtime = supervisor.children["runtime"].process
        assert runtime is not None
        runtime.returncode = 1

        degraded = await supervisor.health()
        assert degraded.status == "degraded"
        assert any(
            child.name == "runtime" and child.status == "down" for child in degraded.children
        )

        await supervisor.supervise_once()
        assert factory.started == ["runtime", "api", "sentinel"]
        # A scheduled restart is reported with its remaining delay.
        scheduled = await supervisor.health()
        runtime_child = next(c for c in scheduled.children if c.name == "runtime")
        assert runtime_child.status == "down"
        assert runtime_child.last_exit_code == 1
        assert "restart_scheduled_in" in (runtime_child.detail or "")
        clock.advance(3.0)
        await supervisor.supervise_once()
        assert factory.started == ["runtime", "api", "sentinel", "runtime"]
        assert supervisor.children["runtime"].restarts == 1
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_supervisor_backoff_is_exponential_capped_and_resets(settings_tmp) -> None:
    factory = FakeFactory()
    clock = ManualClock()
    supervisor = _supervisor(settings_tmp, factory, clock, stable_after_seconds=30.0)
    await supervisor.start()
    try:
        child = supervisor.children["runtime"]
        base = child.spec.restart_backoff_seconds

        # First failure schedules base backoff; restart fires exactly at it.
        assert child.process is not None
        child.process.returncode = 1
        await supervisor.supervise_once()
        assert child.next_restart_at == pytest.approx(clock.now + base)
        clock.advance(base)
        await supervisor.supervise_once()
        assert child.restarts == 1

        # Immediate second failure doubles the delay deterministically.
        assert child.process is not None
        child.process.returncode = 1
        await supervisor.supervise_once()
        assert child.next_restart_at == pytest.approx(clock.now + base * 2)
        clock.advance(base * 2)
        await supervisor.supervise_once()
        assert child.restarts == 2

        # After a long stable run the failure streak resets to base backoff.
        clock.advance(31.0)
        await supervisor.supervise_once()
        assert child.consecutive_failures == 0
        assert child.process is not None
        child.process.returncode = 1
        await supervisor.supervise_once()
        assert child.next_restart_at == pytest.approx(clock.now + base)
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_supervisor_health_gates_dependent_children(settings_tmp) -> None:
    factory = FakeFactory()

    async def unhealthy_runtime(spec) -> bool:
        return spec.name != "runtime"

    supervisor = AprialdSupervisor(
        settings_tmp.model_copy(
            update={
                "daemon": settings_tmp.daemon.model_copy(
                    update={"startup_timeout_seconds": 0.01, "health_poll_seconds": 0.01}
                )
            }
        ),
        process_factory=factory,
        health_checker=unhealthy_runtime,
        governor=_permissive_governor(settings_tmp),
        sleep=_no_sleep,
        clock=ManualClock(),
    )
    await supervisor.start()
    try:
        assert factory.started == ["runtime"]
        assert supervisor.children["api"].process is None
    finally:
        await supervisor.stop()


def test_stop_daemon_stops_any_live_status_and_handles_stale_and_permission(
    settings_tmp,
) -> None:
    pid_path = daemon_pid_path(settings_tmp)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("4242\n", encoding="utf-8")
    alive = {4242: True}
    signals: list[int] = []

    def kill(pid: int, sig: int) -> None:
        assert pid == 4242
        signals.append(sig)
        alive[pid] = False

    stopped = stop_daemon(
        settings_tmp,
        kill=kill,
        pid_alive=lambda pid: alive.get(pid, False),
    )
    assert stopped == {"status": "stopped", "pid": None}
    assert signals == [signal.SIGTERM]

    pid_path.write_text("4243\n", encoding="utf-8")
    assert stop_daemon(settings_tmp, pid_alive=lambda _pid: False)["status"] == "stopped"
    assert not pid_path.exists()

    pid_path.write_text("4244\n", encoding="utf-8")

    def denied(_pid: int, _sig: int) -> None:
        raise PermissionError

    denied_result = stop_daemon(
        settings_tmp,
        kill=denied,
        pid_alive=lambda _pid: True,
    )
    assert denied_result["status"] == "degraded"
    assert denied_result["pid"] == 4244
    assert pid_path.exists()


def test_wait_for_core_health_is_bounded_and_actionable(settings_tmp) -> None:
    clock = ManualClock()

    def advance(seconds: float) -> None:
        clock.advance(seconds)

    with pytest.raises(RuntimeError, match=r"apriald\.status\.json"):
        wait_for_core_health(
            settings_tmp,
            timeout_seconds=0.2,
            probe=lambda _url: False,
            sleep=advance,
            clock=clock,
        )
    assert (
        wait_for_core_health(settings_tmp, probe=lambda _url: True, sleep=advance, clock=clock)[
            "status"
        ]
        == "running"
    )


def test_resource_governor_uses_fake_signals(settings_tmp) -> None:
    governor = ResourceGovernor(
        settings_tmp,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=0.5,
                cpu_load_percent=95.0,
                on_ac_power=False,
                user_idle_seconds=10.0,
            )
        ),
        policy=ResourcePolicy(
            min_ram_headroom_gb=2.0,
            max_cpu_load_percent=80.0,
            require_ac_power_for_background=True,
            min_idle_seconds_for_background=300.0,
        ),
    )
    resident = governor.assess_resident()
    assert resident.allowed is False
    assert resident.reasons == ("ram_headroom_below_policy", "cpu_load_above_policy")

    background = governor.assess_background()
    assert background.allowed is False
    assert "ac_power_required" in background.reasons
    assert "user_not_idle" in background.reasons


def test_launchd_install_writes_only_user_launch_agents(settings_tmp, tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    manager = LaunchdManager(settings_tmp, user_home=user_home)
    path = manager.install()

    assert path == user_home / "Library" / "LaunchAgents" / "com.april.apriald.plist"
    assert manager.status()["installed"] is True
    with path.open("rb") as fh:
        payload = plistlib.load(fh)
    argv = payload["ProgramArguments"]
    assert argv[1:] == ["-m", "apps.daemon.apriald"]
    assert payload["EnvironmentVariables"]["APRIL_HOME"] == str(settings_tmp.home)
    # User LaunchAgent semantics: start at login and keep the supervisor alive.
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is True

    assert manager.uninstall() is True
    assert manager.status()["installed"] is False


def test_launchd_argv_lifecycle_is_idempotent_and_truthful(settings_tmp, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    loaded = False

    def runner(argv) -> subprocess.CompletedProcess[str]:
        nonlocal loaded
        args = list(argv)
        calls.append(args)
        if args[1] == "print":
            return subprocess.CompletedProcess(args, 0 if loaded else 113, "", "")
        if args[1] == "bootstrap":
            loaded = True
        elif args[1] == "bootout":
            loaded = False
        return subprocess.CompletedProcess(args, 0, "", "")

    manager = LaunchdManager(
        settings_tmp,
        user_home=tmp_path / "user",
        runner=runner,
        platform="darwin",
        uid=501,
    )
    assert manager.bootstrap()["error"] == "LaunchAgent plist is not installed"
    manager.install()
    assert manager.bootstrap()["changed"] is True
    assert manager.bootstrap()["changed"] is False
    assert manager.kickstart()["started"] is True
    assert manager.bootout()["changed"] is True
    assert manager.bootout()["changed"] is False
    mutation_calls = [call for call in calls if call[1] != "print"]
    assert mutation_calls == [
        ["launchctl", "bootstrap", "gui/501", str(manager.plist_path)],
        ["launchctl", "kickstart", "-k", "gui/501/com.april.apriald"],
        ["launchctl", "bootout", "gui/501/com.april.apriald"],
    ]

    unsupported = LaunchdManager(
        settings_tmp, user_home=tmp_path / "linux", platform="linux", runner=runner
    )
    assert unsupported.status()["supported"] is False
    assert unsupported.bootstrap()["supported"] is False
