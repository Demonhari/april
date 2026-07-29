from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from april_common.audit import AuditLogger, audit_logger_for_settings
from april_common.process_environment import (
    ProcessCategory,
    build_process_environment,
    without_raw_credentials,
)
from april_common.service_health import probe_service_health
from april_common.settings import AprilSettings, get_settings
from april_common.time import utc_now_iso
from services.pool.governor import GovernorDecision, ResourceGovernor
from services.tool_worker.limits import (
    default_tool_worker_runtime_directory,
    prepare_runtime_directory,
    validate_live_socket,
    write_capability_file,
)


@dataclass(frozen=True, slots=True)
class ChildSpec:
    name: str
    argv: tuple[str, ...]
    health_url: str | None = None
    health_token: str | None = field(default=None, repr=False)
    process_category: ProcessCategory = ProcessCategory.DAEMON
    environment_overrides: tuple[tuple[str, str], ...] = ()
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
    health_failures: int = 0
    crash_loop_suppressed: bool = False


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
        audit: AuditLogger | None = None,
    ) -> None:
        self.settings = settings
        self.process_factory = process_factory or self._spawn_process
        self._uses_default_process_factory = process_factory is None
        self.health_checker = health_checker or _loopback_health_check
        self.governor = governor or ResourceGovernor(settings)
        self.sleep = sleep
        self.clock = clock
        self.stable_after_seconds = stable_after_seconds
        self.audit = audit or audit_logger_for_settings(settings)
        self.lock = DaemonLock(daemon_lock_path(settings))
        self.children: dict[str, ChildRuntime] = {
            spec.name: ChildRuntime(spec=spec) for spec in default_child_specs(settings)
        }
        self._stopped = False

    async def start(self) -> None:
        self.lock.acquire()
        _write_pid_file(self.settings, os.getpid())
        self._audit("daemon_start")
        for runtime in self.children.values():
            await self._ensure_child(runtime)
            if runtime.paused_reason is not None:
                continue
            if not await self._wait_child_ready(runtime):
                self._audit(
                    "daemon_child_dependency_blocked",
                    child=runtime.spec.name,
                    detail="startup health did not become ready",
                )
                break
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
            if runtime.paused_reason is not None:
                continue
            if not await self._dependency_ready(runtime):
                break
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
            self._signal_process(process, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                self._signal_process(process, signal.SIGKILL)
                await process.wait()
            self._audit("daemon_child_exit", child=runtime.spec.name)
        daemon_pid_path(self.settings).unlink(missing_ok=True)
        self._write_stopped_status()
        self.lock.release()
        self._audit("daemon_stop")

    async def _ensure_child(self, runtime: ChildRuntime) -> None:
        decision = self.governor.assess_resident()
        if not decision.allowed:
            runtime.paused_reason = ",".join(decision.reasons)
            return
        runtime.paused_reason = None
        process = runtime.process
        if process is not None and process.returncode is None:
            await self._track_live_health(runtime)
            if process.returncode is not None:
                return
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
                self._audit(
                    "daemon_child_backoff",
                    child=runtime.spec.name,
                    detail=f"failures={runtime.consecutive_failures}",
                )
                return
            if self.clock() < runtime.next_restart_at:
                return
            runtime.restarts += 1
        runtime.process = await self.process_factory(runtime.spec)
        runtime.last_started_at = self.clock()
        runtime.next_restart_at = None
        self._audit("daemon_child_start", child=runtime.spec.name)

    async def _wait_child_ready(self, runtime: ChildRuntime) -> bool:
        """Bound startup verification before a dependent child is launched."""
        if runtime.spec.health_url is None:
            return True
        timeout = self.settings.daemon.startup_timeout_seconds
        poll = max(0.01, self.settings.daemon.health_poll_seconds)
        attempts = max(1, int(timeout / poll) + 1)
        for attempt in range(attempts):
            process = runtime.process
            if process is None or process.returncode is not None:
                return False
            if await self.health_checker(runtime.spec):
                return True
            if attempt + 1 < attempts:
                await self.sleep(poll)
        return False

    async def _dependency_ready(self, runtime: ChildRuntime) -> bool:
        process = runtime.process
        if process is None or process.returncode is not None:
            return False
        if runtime.spec.health_url is None:
            return True
        return await self.health_checker(runtime.spec)

    async def _track_live_health(self, runtime: ChildRuntime) -> None:
        if runtime.spec.health_url is None or runtime.process is None:
            return
        if (
            runtime.last_started_at is not None
            and self.clock() - runtime.last_started_at
            < self.settings.daemon.child_startup_grace_seconds
        ):
            return
        healthy = await self.health_checker(runtime.spec)
        if healthy:
            if runtime.health_failures or runtime.crash_loop_suppressed:
                self._audit("daemon_child_recovered", child=runtime.spec.name)
            runtime.health_failures = 0
            runtime.crash_loop_suppressed = False
            return
        runtime.health_failures += 1
        self._audit(
            "daemon_child_health_failure",
            child=runtime.spec.name,
            detail=f"consecutive={runtime.health_failures}",
        )
        if runtime.health_failures < self.settings.daemon.child_health_failure_threshold:
            return
        if runtime.restarts >= self.settings.daemon.child_crash_loop_threshold:
            runtime.crash_loop_suppressed = True
            self._audit("daemon_child_crash_loop_suppressed", child=runtime.spec.name)
            return
        self._signal_process(runtime.process, signal.SIGTERM)

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
                    f"restart_scheduled_in={remaining:.1f}s failures={runtime.consecutive_failures}"
                )
            return ChildHealth(
                runtime.spec.name,
                "down",
                process.pid,
                runtime.restarts,
                last_exit_code=process.returncode,
                detail=detail,
            )
        if runtime.crash_loop_suppressed:
            return ChildHealth(
                runtime.spec.name,
                "degraded",
                process.pid,
                runtime.restarts,
                detail="crash_loop_suppressed",
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
        env = build_process_environment(
            spec.process_category,
            source=without_raw_credentials(),
            april_home=self.settings.home,
            overrides=dict(spec.environment_overrides),
        )
        # Passing asyncio.subprocess.DEVNULL asks asyncio to create parent-side
        # file objects whose cleanup otherwise depends on transport GC on some
        # supported Python/macOS combinations. Own the handle explicitly so
        # every spawn has deterministic resource lifetime.
        with Path(os.devnull).open("wb") as devnull:
            return await asyncio.create_subprocess_exec(
                *spec.argv,
                cwd=str(self.settings.home),
                env=env,
                stdout=devnull,
                stderr=devnull,
                start_new_session=True,
            )

    def _signal_process(self, process: ProcessHandle, signum: int) -> None:
        if self._uses_default_process_factory:
            try:
                os.killpg(process.pid, signum)
                return
            except ProcessLookupError:
                return
        if signum == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()

    def _audit(
        self, event_type: str, *, child: str | None = None, detail: str | None = None
    ) -> None:
        self.audit.write(
            {
                "event_type": event_type,
                "actor": "apriald",
                "child": child,
                "detail": detail,
            }
        )


def default_child_specs(settings: AprilSettings) -> tuple[ChildSpec, ...]:
    python = sys.executable
    specs = [
        ChildSpec(
            name="runtime",
            argv=(python, "-m", "services.april_runtime.server"),
            health_url=settings.runtime.url.rstrip("/") + "/runtime/health",
            health_token=settings.runtime.token,
            process_category=ProcessCategory.RUNTIME,
        ),
    ]
    external_overrides: list[tuple[str, str]] = []
    if settings.workers.tool_worker_enabled:
        tool_runtime = prepare_runtime_directory(
            default_tool_worker_runtime_directory(settings.home),
            april_home=settings.home,
        )
        tool_socket = tool_runtime / "worker.sock"
        capability_file = tool_runtime / "capability"
        write_capability_file(
            capability_file,
            secrets.token_urlsafe(32),
            runtime_directory=tool_runtime,
        )
        specs.append(
            ChildSpec(
                name="tool_worker",
                argv=(
                    python,
                    "-m",
                    "services.tool_worker.server",
                    "--april-home",
                    str(settings.home),
                    "--socket",
                    str(tool_socket),
                    "--capability-file",
                    str(capability_file),
                    *tuple(
                        part
                        for root in settings.allowed_roots
                        for part in ("--allowed-root", str(root))
                    ),
                ),
                health_url=f"tool-worker://{tool_socket}",
                process_category=ProcessCategory.TOOL_WORKER,
            )
        )
        external_overrides.append(("APRIL_TOOL_WORKER_EXTERNAL", "1"))
    if settings.workers.job_worker_enabled:
        job_runtime = settings.home / "data" / "runtime" / "job-worker"
        job_runtime.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(job_runtime, 0o700)
        specs.append(
            ChildSpec(
                name="job_worker",
                argv=(
                    python,
                    "-m",
                    "services.jobs.worker",
                    "--april-home",
                    str(settings.home),
                    "--status-file",
                    str(job_runtime / "status.json"),
                ),
                health_url=f"job-worker://{job_runtime / 'status.json'}",
                process_category=ProcessCategory.JOB_WORKER,
            )
        )
        external_overrides.append(("APRIL_JOB_WORKER_EXTERNAL", "1"))
    specs.append(
        ChildSpec(
            name="api",
            argv=(python, "-m", "services.api.server"),
            health_url=f"http://{settings.api.host}:{settings.api.port}/health",
            process_category=ProcessCategory.CORE_API,
            environment_overrides=tuple(external_overrides),
        )
    )
    # The Sentinel needs a microphone, wake models, and STT. Supervising it with
    # voice or wake disabled would only crash-loop (run_sentinel raises), so the
    # default safe config supervises runtime and API only.
    if settings.voice.enabled and settings.wake.enabled:
        from services.wake.control import sentinel_control_path

        specs.append(
            ChildSpec(
                name="sentinel",
                argv=(python, "-m", "services.wake.sentinel"),
                health_url=f"sentinel-control://{sentinel_control_path(settings)}",
                process_category=ProcessCategory.SENTINEL_VOICE,
            )
        )
    return tuple(specs)


async def _loopback_health_check(spec: ChildSpec) -> bool:
    if spec.health_url is None:
        return True
    if spec.health_url.startswith("sentinel-control://"):
        try:
            from services.wake.control import sentinel_status_at_path

            # The IPC reports actual Sentinel state, including mute/listener and
            # degraded voice output; process existence alone is insufficient.
            path = Path(spec.health_url.removeprefix("sentinel-control://"))
            status = await asyncio.to_thread(sentinel_status_at_path, path)
            return status.get("state") in {"listening", "muted"}
        except (OSError, RuntimeError, ValueError):
            return False
    if spec.health_url.startswith("tool-worker://"):
        try:
            path = Path(spec.health_url.removeprefix("tool-worker://"))
            validate_live_socket(path, runtime_directory=path.parent)
            return True
        except (OSError, RuntimeError):
            return False
    if spec.health_url.startswith("job-worker://"):
        try:
            path = Path(spec.health_url.removeprefix("job-worker://"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("version") == 1 and payload.get("ready") is True
        except (OSError, ValueError, TypeError):
            return False
    result = await asyncio.to_thread(
        probe_service_health,
        spec.health_url,
        bearer_token=spec.health_token,
        timeout=1.0,
    )
    return result.ok


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
        # Only the stale ownership hint is removed; historical status remains
        # available for diagnosis and the next start can proceed safely.
        pid_path.unlink(missing_ok=True)
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


def wait_for_core_health(
    settings: AprilSettings,
    *,
    timeout_seconds: float | None = None,
    probe: Callable[[str], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Boundedly wait for the loopback Core API used by every startup path."""
    url = f"http://{settings.api.host}:{settings.api.port}/health"
    if settings.api.host not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("Daemon health polling is restricted to loopback.")
    active_probe = probe or _sync_health_probe
    timeout = timeout_seconds or settings.daemon.startup_timeout_seconds
    deadline = clock() + timeout
    while True:
        if active_probe(url):
            return {"status": "running", "health_url": url}
        if clock() >= deadline:
            log_path = settings.logs_path / "apriald.log"
            raise RuntimeError(
                f"APRIL Core API did not become healthy within {timeout:.1f}s; "
                f"inspect {log_path} and {daemon_status_path(settings)}"
            )
        sleep(min(settings.daemon.health_poll_seconds, max(0.0, deadline - clock())))


def start_daemon_background(
    settings: AprilSettings,
    *,
    health_probe: Callable[[str], bool] | None = None,
) -> dict[str, object]:
    current = read_daemon_status(settings)
    current_pid = current.get("pid")
    if isinstance(current_pid, int) and _pid_alive(current_pid):
        wait_for_core_health(settings, probe=health_probe)
        return current
    settings.logs_path.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_path / "apriald.log"
    env = build_process_environment(
        ProcessCategory.DAEMON,
        source=without_raw_credentials(),
        april_home=settings.home,
    )
    with log_path.open("ab") as log_file, Path(os.devnull).open("rb") as devnull:
        process = subprocess.Popen(
            [sys.executable, "-m", "apps.daemon.apriald"],
            cwd=str(settings.home),
            env=env,
            stdin=devnull,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    _write_pid_file(settings, process.pid)
    try:
        wait_for_core_health(settings, probe=health_probe)
    except Exception:
        _write_status_payload(
            settings,
            {
                "schema_version": 1,
                "status": "degraded",
                "pid": process.pid,
                "generated_at": utc_now_iso(),
                "children": [],
                "governor": {"allowed": None, "reasons": []},
            },
        )
        raise
    return {"status": "running", "pid": process.pid, "log_path": str(log_path)}


def stop_daemon(
    settings: AprilSettings,
    *,
    kill: Callable[[int, int], None] = os.kill,
    pid_alive: Callable[[int], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    active_pid_alive = pid_alive or _pid_alive
    pid_path = daemon_pid_path(settings)
    if not pid_path.exists():
        return {"status": "stopped", "pid": None}
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pid_path.unlink(missing_ok=True)
        return {"status": "stale", "pid": None, "detail": "invalid pid file removed"}
    if not active_pid_alive(pid):
        pid_path.unlink(missing_ok=True)
        return {"status": "stopped", "pid": None}
    try:
        kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return {"status": "stopped", "pid": None}
    except PermissionError:
        return {"status": "degraded", "pid": pid, "detail": "permission denied stopping PID"}
    deadline = clock() + settings.daemon.shutdown_timeout_seconds
    while active_pid_alive(pid) and clock() < deadline:
        sleep(min(0.1, max(0.0, deadline - clock())))
    if active_pid_alive(pid):
        # Local documented fallback after the bounded graceful period.
        try:
            kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            return {
                "status": "degraded",
                "pid": pid,
                "detail": "permission denied forcing stopped PID",
            }
    pid_path.unlink(missing_ok=True)
    return {"status": "stopped", "pid": None}


def autostart_if_needed(settings: AprilSettings) -> dict[str, object]:
    status = read_daemon_status(settings)
    pid = status.get("pid")
    if isinstance(pid, int) and _pid_alive(pid):
        wait_for_core_health(settings)
        return status
    return start_daemon_background(settings)


def _sync_health_probe(url: str) -> bool:
    return probe_service_health(url, timeout=1.0).ok


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
        and all(isinstance(child, dict) and child.get("status") == "paused" for child in children)
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
