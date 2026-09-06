from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol
from urllib.parse import urlparse

from april_common.process_environment import (
    ProcessCategory,
    build_process_environment,
    without_raw_credentials,
)
from april_common.service_health import probe_service_health
from april_common.settings import AprilSettings, load_settings, project_root


class PopenFactory(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        stdout: IO[bytes],
        stderr: int,
        start_new_session: bool,
    ) -> subprocess.Popen[bytes]: ...


HealthGetter = Callable[[str, float], bool]
PidExists = Callable[[int], bool]
PidSignal = Callable[[int, int], None]
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class ServiceInfo:
    name: str
    pid: int | None
    running: bool
    healthy: bool
    log_path: Path
    backend: str | None = None


@dataclass(frozen=True)
class ServiceStatus:
    runtime: ServiceInfo
    api: ServiceInfo
    owner: str = "unknown"
    backend: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.runtime.running and self.runtime.healthy and self.api.running and self.api.healthy
        )


class AprilServiceManager:
    def __init__(
        self,
        *,
        home: Path | None = None,
        python_executable: str | None = None,
        popen_factory: PopenFactory | None = None,
        health_getter: HealthGetter | None = None,
        pid_exists: PidExists | None = None,
        pid_signal: PidSignal | None = None,
        sleep: Sleeper = time.sleep,
        startup_timeout_seconds: float = 20.0,
    ) -> None:
        self.home = self._locate_home(home)
        os.environ.setdefault("APRIL_HOME", str(self.home))
        self.settings = self._settings_for_home(self.home)
        self.python_executable = python_executable or sys.executable
        self.popen_factory = popen_factory or subprocess.Popen
        self.health_getter = health_getter or self._authenticated_health_getter
        self.pid_exists = pid_exists or self._default_pid_exists
        self.pid_signal = pid_signal or os.kill
        self.sleep = sleep
        self.startup_timeout_seconds = startup_timeout_seconds
        self.run_dir = self.home / "data" / "run"
        self.log_dir = self.home / "logs"
        self.runtime_pid_path = self.run_dir / "runtime.pid"
        self.api_pid_path = self.run_dir / "api.pid"
        self.runtime_log_path = self.log_dir / "runtime.log"
        self.api_log_path = self.log_dir / "api.log"
        self.lifecycle_path = self.run_dir / "lifecycle.json"

    def status(self) -> ServiceStatus:
        lifecycle = self._read_lifecycle()
        owner = str(lifecycle.get("owner", "unknown")) if lifecycle else "unknown"
        backend = str(lifecycle.get("backend")) if lifecycle and lifecycle.get("backend") else None
        status = ServiceStatus(
            runtime=self._service_info(
                name="runtime",
                pid_path=self.runtime_pid_path,
                log_path=self.runtime_log_path,
                health_url=f"{self.settings.runtime.url}/runtime/health",
            ),
            api=self._service_info(
                name="api",
                pid_path=self.api_pid_path,
                log_path=self.api_log_path,
                health_url=f"http://{self.settings.api.host}:{self.settings.api.port}/health",
            ),
            owner=owner,
            backend=backend,
        )
        if not status.runtime.running and not status.api.running and owner == "runner":
            self.lifecycle_path.unlink(missing_ok=True)
            status = ServiceStatus(status.runtime, status.api, "unknown", None)
        if status.owner == "unknown":
            daemon = self._daemon_children()
            if daemon is not None:
                backend, pids = daemon
                runtime = self._service_info_for_pid(
                    "runtime",
                    pids.get("runtime"),
                    self.runtime_log_path,
                    f"{self.settings.runtime.url}/runtime/health",
                )
                api = self._service_info_for_pid(
                    "api",
                    pids.get("api"),
                    self.api_log_path,
                    f"http://{self.settings.api.host}:{self.settings.api.port}/health",
                )
                return ServiceStatus(runtime, api, "daemon", backend)
        return status

    def start(self, *, fake_backend: bool = False) -> ServiceStatus:
        self._ensure_dirs()
        current = self.status()
        requested_backend = "fake" if fake_backend else "real"
        if current.backend is not None and current.backend != requested_backend:
            raise RuntimeError(
                f"APRIL services are already owned by {current.owner} in {current.backend} mode; "
                f"refusing incompatible {requested_backend} reuse."
            )
        if current.owner == "daemon" and current.ok:
            return current
        if not current.runtime.running:
            self._start_service(
                pid_path=self.runtime_pid_path,
                log_path=self.runtime_log_path,
                module="services.april_runtime.server",
                fake_backend=fake_backend,
            )
        self._wait_for_health(f"{self.settings.runtime.url}/runtime/health")

        current = self.status()
        if not current.api.running:
            self._start_service(
                pid_path=self.api_pid_path,
                log_path=self.api_log_path,
                module="services.api.server",
                fake_backend=fake_backend,
            )
        self._wait_for_health(f"http://{self.settings.api.host}:{self.settings.api.port}/health")
        result = self.status()
        self._write_lifecycle(owner="runner", backend=requested_backend, status=result)
        return self.status()

    def stop(self) -> ServiceStatus:
        current = self.status()
        if current.owner == "daemon":
            from apps.daemon.apriald import stop_daemon

            stop_daemon(self.settings)
            self.lifecycle_path.unlink(missing_ok=True)
            return self.status()
        self._stop_service(self.api_pid_path)
        self._stop_service(self.runtime_pid_path)
        self.lifecycle_path.unlink(missing_ok=True)
        return self.status()

    def stop_started_services(self, before: ServiceStatus) -> ServiceStatus:
        """Stop only services absent before a one-shot invocation started."""
        if not before.api.running:
            self._stop_service(self.api_pid_path)
        if not before.runtime.running:
            self._stop_service(self.runtime_pid_path)
        if not self.status().runtime.running and not self.status().api.running:
            self.lifecycle_path.unlink(missing_ok=True)
        return self.status()

    def restart(self, *, fake_backend: bool = False) -> ServiceStatus:
        self.stop()
        return self.start(fake_backend=fake_backend)

    def _read_lifecycle(self) -> dict[str, object] | None:
        try:
            payload = json.loads(self.lifecycle_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _daemon_children(self) -> tuple[str, dict[str, int]] | None:
        try:
            from apps.daemon.apriald import daemon_pid_path, daemon_status_path

            daemon_pid = int(daemon_pid_path(self.settings).read_text(encoding="utf-8"))
            if not self.pid_exists(daemon_pid):
                return None
            payload = json.loads(daemon_status_path(self.settings).read_text(encoding="utf-8"))
            children = payload.get("children", [])
            pids = {
                str(child["name"]): int(child["pid"])
                for child in children
                if isinstance(child, dict) and child.get("pid") is not None
            }
            return (str(payload.get("runtime_backend", self.settings.runtime.backend)), pids)
        except (
            ImportError,
            FileNotFoundError,
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            return None

    def _write_lifecycle(self, *, owner: str, backend: str, status: ServiceStatus) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.lifecycle_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "owner": owner,
                    "backend": backend,
                    "runtime_pid": status.runtime.pid,
                    "api_pid": status.api.pid,
                    "updated_at": time.time(),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.lifecycle_path)

    def logs(self, *, lines: int = 80) -> str:
        capped_lines = max(1, min(lines, 1000))
        chunks = [
            f"== April Runtime: {self.runtime_log_path} ==",
            self._tail(self.runtime_log_path, capped_lines),
            f"== April Core API: {self.api_log_path} ==",
            self._tail(self.api_log_path, capped_lines),
        ]
        return "\n".join(chunks)

    def _start_service(
        self,
        *,
        pid_path: Path,
        log_path: Path,
        module: str,
        fake_backend: bool,
    ) -> None:
        category = (
            ProcessCategory.RUNTIME
            if module == "services.april_runtime.server"
            else ProcessCategory.CORE_API
        )
        env = self._child_env(fake_backend=fake_backend, category=category)
        args = [self.python_executable, "-m", module]
        with log_path.open("ab") as log_file:
            process = self.popen_factory(
                args,
                cwd=str(self.home),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(str(process.pid), encoding="utf-8")

    def _stop_service(self, pid_path: Path) -> None:
        pid = self._read_pid(pid_path)
        if pid is None or not self.pid_exists(pid):
            pid_path.unlink(missing_ok=True)
            return
        self.pid_signal(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not self.pid_exists(pid):
                pid_path.unlink(missing_ok=True)
                return
            self.sleep(0.1)
        if self.pid_exists(pid):
            self.pid_signal(pid, signal.SIGKILL)
        pid_path.unlink(missing_ok=True)

    def _wait_for_health(self, url: str) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.health_getter(url, 1.0):
                return
            self.sleep(0.2)
        parsed = urlparse(url)
        port = parsed.port
        port_detail = ""
        if port is not None and self._port_in_use(parsed.hostname or "127.0.0.1", port):
            port_detail = (
                f"\nPort {port} is in use. Run `run april status`, inspect logs with "
                "`run april logs --tail 100`, or stop stale services with `run april stop`."
            )
        raise RuntimeError(
            "APRIL service did not become healthy in time."
            + port_detail
            + "\n\n"
            + self.logs(lines=40)
        )

    def _service_info(
        self,
        *,
        name: str,
        pid_path: Path,
        log_path: Path,
        health_url: str,
    ) -> ServiceInfo:
        pid = self._read_pid(pid_path)
        if pid is None:
            return ServiceInfo(name=name, pid=None, running=False, healthy=False, log_path=log_path)
        if not self.pid_exists(pid):
            pid_path.unlink(missing_ok=True)
            return ServiceInfo(name=name, pid=None, running=False, healthy=False, log_path=log_path)
        healthy = self.health_getter(health_url, 1.0)
        return ServiceInfo(name=name, pid=pid, running=True, healthy=healthy, log_path=log_path)

    def _service_info_for_pid(
        self, name: str, pid: int | None, log_path: Path, health_url: str
    ) -> ServiceInfo:
        if pid is None or not self.pid_exists(pid):
            return ServiceInfo(name, None, False, False, log_path)
        return ServiceInfo(name, pid, True, self.health_getter(health_url, 1.0), log_path)

    def _read_pid(self, pid_path: Path) -> int | None:
        try:
            raw = pid_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        try:
            return int(raw)
        except ValueError:
            pid_path.unlink(missing_ok=True)
            return None

    def _child_env(
        self,
        *,
        fake_backend: bool,
        category: ProcessCategory = ProcessCategory.CORE_API,
    ) -> dict[str, str]:
        env = build_process_environment(
            category,
            source=without_raw_credentials(),
            april_home=self.home,
        )
        if fake_backend and category is ProcessCategory.RUNTIME:
            env["APRIL_RUNTIME_BACKEND"] = "fake"
        return env

    def _ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _tail(self, path: Path, lines: int) -> str:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            return "(no log file)"
        return "\n".join(content[-lines:]) if content else "(empty)"

    def _settings_for_home(self, home: Path) -> AprilSettings:
        return load_settings(root=home)

    def _locate_home(self, explicit: Path | None) -> Path:
        if explicit is not None:
            return explicit.expanduser().resolve()
        if os.environ.get("APRIL_HOME"):
            return Path(os.environ["APRIL_HOME"]).expanduser().resolve()
        root = project_root()
        if (root / "pyproject.toml").exists():
            return root
        return Path.cwd().resolve()

    @staticmethod
    def _default_pid_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _default_health_getter(url: str, timeout: float) -> bool:
        return probe_service_health(url, timeout=timeout).ok

    def _authenticated_health_getter(self, url: str, timeout: float) -> bool:
        token = self.settings.runtime.token if url.startswith(self.settings.runtime.url) else None
        return probe_service_health(url, bearer_token=token, timeout=timeout).ok

    def _port_in_use(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((host, port)) == 0
