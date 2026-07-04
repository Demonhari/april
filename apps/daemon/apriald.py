from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from april_common.settings import AprilSettings, get_settings
from april_common.time import utc_now_iso
from services.pool.governor import GovernorDecision, ResourceGovernor


@dataclass(frozen=True, slots=True)
class ChildSpec:
    name: str
    argv: tuple[str, ...]
    health_url: str | None = None
    restart_backoff_seconds: float = 2.0
    max_restart_backoff_seconds: float = 60.0


@dataclass(slots=True)
class ChildRuntime:
    spec: ChildSpec
    process: ProcessHandle | None = None
    restarts: int = 0
    consecutive_failures: int = 0
    next_restart_at: float | None = None
    last_started_at: float | None = None
    paused_reason: str | None = None
    last_exit_code: int | None = None


@dataclass(frozen=True, slots=True)
class ChildHealth:
    name: str
    status: str
    pid: int | None
    restarts: int
    last_exit_code: int | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DaemonHealth:
    status: str
    children: tuple[ChildHealth, ...]
    governor: GovernorDecision


class ProcessHandle(Protocol):
    @property
    def pid(self) -> int: ...

    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


ProcessFactory = Callable[[ChildSpec], Awaitable[ProcessHandle]]
HealthChecker = Callable[[ChildSpec], Awaitable[bool]]
Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


def daemon_lock_path(settings: AprilSettings) -> Path:
    return settings.resolve_path(Path("data/apriald.lock"))


def daemon_pid_path(settings: AprilSettings) -> Path:
    return settings.resolve_path(Path("data/apriald.pid"))


def daemon_status_path(settings: AprilSettings) -> Path:
    return settings.resolve_path(Path("data/apriald.status.json"))


class DaemonLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        import fcntl

        if self._fd is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError("apriald is already running.") from exc
        self._fd = fd

    def release(self) -> None:
        import fcntl

        fd = self._fd
        self._fd = None
        if fd is None:
            return
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class AprialdSupervisor:
    def __init__(
        self,
        settings: AprilSettings,
        *,
        process_factory: ProcessFactory | None = None,
        health_checker: HealthChecker | None = None,
        governor: ResourceGovernor | None = None,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
        stable_after_seconds: float = 30.0,
    ) -> None:
        self.settings = settings
        self.process_factory = process_factory or self._spawn_process
        self.health_checker = health_checker or _loopback_health_check
        self.governor = governor or ResourceGovernor(settings)
        self.sleep = sleep
        self.clock = clock
        self.stable_after_seconds = stable_after_seconds
        self.lock = DaemonLock(daemon_lock_path(settings))
        self.children: dict[str, ChildRuntime] = {
            spec.name: ChildRuntime(spec=spec) for spec in default_child_specs(settings)
        }
        self._stopped = False

    async def start(self) -> None:
        self.lock.acquire()
        _write_pid_file(self.settings, os.getpid())
        for runtime in self.children.values():
            await self._ensure_child(runtime)
        self._write_status(await self.health())

    async def run_forever(self, *, interval_seconds: float = 2.0) -> None:
        await self.start()
        try:
            while not self._stopped:
                await self.supervise_once()
                await self.sleep(interval_seconds)
        finally:
            await self.stop()

    async def supervise_once(self) -> DaemonHealth:
        for runtime in self.children.values():
            await self._ensure_child(runtime)
        health = await self.health()
        self._write_status(health)
        return health

    async def health(self) -> DaemonHealth:
        governor = self.governor.assess_resident()
        child_health: list[ChildHealth] = []
        degraded = not governor.allowed
        for runtime in self.children.values():
            child = await self._child_health(runtime)
            child_health.append(child)
            if child.status != "running":
                degraded = True
        status = "paused" if not governor.allowed else ("degraded" if degraded else "ok")
        return DaemonHealth(
            status=status,
            children=tuple(child_health),
            governor=governor,
        )

    async def stop(self) -> None:
        self._stopped = True
        for runtime in self.children.values():
            process = runtime.process
            if process is None or process.returncode is not None:
                continue
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        daemon_pid_path(self.settings).unlink(missing_ok=True)
        self._write_stopped_status()
        self.lock.release()

    async def _ensure_child(self, runtime: ChildRuntime) -> None:
        decision = self.governor.assess_resident()
        if not decision.allowed:
            runtime.paused_reason = ",".join(decision.reasons)
            return
        runtime.paused_reason = None
        process = runtime.process
        if process is not None and process.returncode is None:
            if (
                runtime.consecutive_failures > 0
                and runtime.last_started_at is not None
                and self.clock() - runtime.last_started_at >= self.stable_after_seconds
            ):
                # The child stayed up long enough: forget the failure streak so
                # the next crash starts again from the base backoff.
                runtime.consecutive_failures = 0
            return
        if process is not None and process.returncode is not None:
            runtime.last_exit_code = process.returncode
            if runtime.next_restart_at is None:
                # Deterministic exponential backoff, doubling per consecutive
                # failure and capped, driven only by the injected clock.
                runtime.consecutive_failures += 1
                delay = min(
                    runtime.spec.restart_backoff_seconds
                    * (2 ** (runtime.consecutive_failures - 1)),
                    runtime.spec.max_restart_backoff_seconds,
                )
                runtime.next_restart_at = self.clock() + delay
                return
            if self.clock() < runtime.next_restart_at:
                return
            runtime.restarts += 1
        runtime.process = await self.process_factory(runtime.spec)
        runtime.last_started_at = self.clock()
        runtime.next_restart_at = None

    async def _child_health(self, runtime: ChildRuntime) -> ChildHealth:
        process = runtime.process
        if runtime.paused_reason is not None:
            return ChildHealth(
                name=runtime.spec.name,
                status="paused",
                pid=process.pid if process is not None and process.returncode is None else None,
                restarts=runtime.restarts,
                last_exit_code=runtime.last_exit_code,
                detail=runtime.paused_reason,
            )
        if process is None:
            return ChildHealth(runtime.spec.name, "stopped", None, runtime.restarts)
        if process.returncode is not None:
            detail = None
            if runtime.next_restart_at is not None:
                remaining = max(0.0, runtime.next_restart_at - self.clock())
                detail = (
                    f"restart_scheduled_in={remaining:.1f}s "
                    f"failures={runtime.consecutive_failures}"
                )
            return ChildHealth(
                runtime.spec.name,
                "down",
                process.pid,
                runtime.restarts,
                last_exit_code=process.returncode,
                detail=detail,
            )
        if runtime.spec.health_url is not None and not await self.health_checker(runtime.spec):
            return ChildHealth(
                runtime.spec.name,
                "degraded",
                process.pid,
                runtime.restarts,
                detail="health_check_failed",
            )
        return ChildHealth(runtime.spec.name, "running", process.pid, runtime.restarts)

    def _write_status(self, health: DaemonHealth) -> None:
        _write_status_payload(self.settings, _daemon_health_payload(health, pid=os.getpid()))

    def _write_stopped_status(self) -> None:
        children = [
            {
                "name": runtime.spec.name,
                "status": "stopped",
                "pid": None,
                "restarts": runtime.restarts,
                "last_exit_code": runtime.last_exit_code,
                "detail": None,
                "paused_reason": None,
                "degraded_reason": None,
            }
            for runtime in self.children.values()
        ]
        _write_status_payload(
            self.settings,
            {
                "schema_version": 1,
                "status": "stopped",
                "pid": None,
                "generated_at": utc_now_iso(),
                "children": children,
                "governor": {"allowed": None, "reasons": []},
            },
        )

    async def _spawn_process(self, spec: ChildSpec) -> ProcessHandle:
        env = os.environ.copy()
        env["APRIL_HOME"] = str(self.settings.home)
        return await asyncio.create_subprocess_exec(
            *spec.argv,
            cwd=str(self.settings.home),
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )


def default_child_specs(settings: AprilSettings) -> tuple[ChildSpec, ...]:
    python = sys.executable
    specs = [
        ChildSpec(
            name="runtime",
            argv=(python, "-m", "services.april_runtime.server"),
            health_url=settings.runtime.url.rstrip("/") + "/health",
        ),
        ChildSpec(
            name="api",
            argv=(python, "-m", "services.api.server"),
            health_url=f"http://{settings.api.host}:{settings.api.port}/health",
        ),
    ]
    # The Sentinel needs a microphone, wake models, and STT. Supervising it with
    # voice or wake disabled would only crash-loop (run_sentinel raises), so the
    # default safe config supervises runtime and API only.
    if settings.voice.enabled and settings.wake.enabled:
        specs.append(
            ChildSpec(
                name="sentinel",
                argv=(python, "-m", "services.wake.sentinel"),
                health_url=None,
            )
        )
    return tuple(specs)


async def _loopback_health_check(spec: ChildSpec) -> bool:
    if spec.health_url is None:
        return True
    try:
        import httpx

        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get(spec.health_url)
        return response.status_code == 200
    except Exception:
        return False


def read_daemon_status(settings: AprilSettings) -> dict[str, object]:
    status_payload = _read_status_payload(settings)
    pid_path = daemon_pid_path(settings)
    if not pid_path.exists():
        return _merge_daemon_status(
            base={"status": "stopped", "pid": None},
            payload=status_payload,
        )
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return _merge_daemon_status(
            base={"status": "degraded", "pid": None, "detail": "invalid pid file"},
            payload=status_payload,
        )
    if not _pid_alive(pid):
        return _merge_daemon_status(
            base={
                "status": "stale",
                "pid": pid,
                "detail": "pid file points to a stopped process",
            },
            payload=status_payload,
        )
    if status_payload is None:
        return {"status": "running", "pid": pid, "details_available": False}
    return _merge_daemon_status(
        base={"status": _operator_status(status_payload), "pid": pid},
        payload=status_payload,
    )


def start_daemon_background(settings: AprilSettings) -> dict[str, object]:
    current = read_daemon_status(settings)
    if current["status"] == "running":
        return current
    settings.logs_path.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_path / "apriald.log"
    log_file = log_path.open("ab")
    env = os.environ.copy()
    env["APRIL_HOME"] = str(settings.home)
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "apps.daemon.apriald"],
            cwd=str(settings.home),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    finally:
        log_file.close()
    _write_pid_file(settings, process.pid)
    return {"status": "starting", "pid": process.pid, "log_path": str(log_path)}


def stop_daemon(settings: AprilSettings) -> dict[str, object]:
    status = read_daemon_status(settings)
    pid = status.get("pid")
    if not isinstance(pid, int):
        return status
    if status["status"] == "running":
        os.kill(pid, signal.SIGTERM)
    daemon_pid_path(settings).unlink(missing_ok=True)
    return {"status": "stopping", "pid": pid}


def autostart_if_needed(settings: AprilSettings) -> None:
    if read_daemon_status(settings)["status"] != "running":
        start_daemon_background(settings)


def _write_pid_file(settings: AprilSettings, pid: int) -> None:
    path = daemon_pid_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")


def _daemon_health_payload(health: DaemonHealth, *, pid: int | None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": health.status,
        "pid": pid,
        "generated_at": utc_now_iso(),
        "children": [_child_health_payload(child) for child in health.children],
        "governor": {
            "allowed": health.governor.allowed,
            "reasons": list(health.governor.reasons),
        },
    }


def _child_health_payload(child: ChildHealth) -> dict[str, object]:
    return {
        "name": child.name,
        "status": child.status,
        "pid": child.pid,
        "restarts": child.restarts,
        "last_exit_code": child.last_exit_code,
        "detail": child.detail,
        "paused_reason": child.detail if child.status == "paused" else None,
        "degraded_reason": child.detail if child.status == "degraded" else None,
    }


def _write_status_payload(settings: AprilSettings, payload: dict[str, object]) -> None:
    path = daemon_status_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    tmp_path.replace(path)


def _read_status_payload(settings: AprilSettings) -> dict[str, Any] | None:
    path = daemon_status_path(settings)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != 1:
        return None
    return payload


def _merge_daemon_status(
    *, base: dict[str, object], payload: dict[str, Any] | None
) -> dict[str, object]:
    merged = dict(base)
    merged["details_available"] = payload is not None
    if payload is None:
        return merged
    for key in ("generated_at", "children", "governor"):
        if key in payload:
            merged[key] = payload[key]
    if "supervisor_status" not in merged and "status" in payload:
        merged["supervisor_status"] = payload["status"]
    return merged


def _operator_status(payload: dict[str, Any]) -> str:
    children = payload.get("children")
    governor = payload.get("governor")
    if (
        isinstance(governor, dict)
        and governor.get("allowed") is False
        and isinstance(children, list)
        and children
        and all(
            isinstance(child, dict) and child.get("status") == "paused"
            for child in children
        )
    ):
        return "paused"
    status = payload.get("status")
    if status == "ok":
        return "running"
    if isinstance(status, str):
        return status
    return "running"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _install_signal_handlers(supervisor: AprialdSupervisor) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: setattr(supervisor, "_stopped", True))


async def _amain() -> None:
    supervisor = AprialdSupervisor(get_settings())
    _install_signal_handlers(supervisor)
    await supervisor.run_forever()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
