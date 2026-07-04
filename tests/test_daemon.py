from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from apps.daemon.apriald import AprialdSupervisor, DaemonLock
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


def _supervisor(
    settings, factory: FakeFactory, clock: ManualClock, **kwargs
) -> AprialdSupervisor:
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
            child.name == "runtime" and child.status == "down"
            for child in degraded.children
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
