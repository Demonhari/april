from __future__ import annotations

import os
import plistlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from april_common.process_environment import ProcessCategory
from april_common.process_runner import RestrictedProcessResult, run_restricted_process_sync
from april_common.settings import AprilSettings

# Final APRIL product LaunchAgent label. The vendor prefix is the project name
# ("april"), not the developer's personal handle, so the daemon identity is
# stable across machines and users. Some early architecture notes referred to a
# "com.hari.apriald" placeholder; "com.april.apriald" is the authoritative label
# used by the installer, the plist filename, and the daemon tests.
LABEL = "com.april.apriald"


class _LaunchResult(Protocol):
    returncode: int | None


class LaunchdManager:
    def __init__(
        self,
        settings: AprilSettings,
        *,
        user_home: Path | None = None,
        runner: Callable[[Sequence[str]], _LaunchResult] | None = None,
        platform: str | None = None,
        uid: int | None = None,
    ) -> None:
        self.settings = settings
        self.user_home = (user_home or Path.home()).expanduser().resolve()
        self.runner = runner or self._run
        self.platform = platform or sys.platform
        self.uid = os.getuid() if uid is None else uid

    @property
    def launch_agents_dir(self) -> Path:
        return self.user_home / "Library" / "LaunchAgents"

    @property
    def plist_path(self) -> Path:
        return self.launch_agents_dir / f"{LABEL}.plist"

    def install(self) -> Path:
        path = self.plist_path
        self._validate_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "Label": LABEL,
            "ProgramArguments": [sys.executable, "-m", "apps.daemon.apriald"],
            "WorkingDirectory": str(self.settings.home),
            "RunAtLoad": True,
            # apriald is the supervisor: launchd must keep it alive so the
            # supervised children (runtime/API/Sentinel) come back after crashes
            # or logout/login. apriald's own lock file keeps it single-instance.
            "KeepAlive": True,
            "EnvironmentVariables": {"APRIL_HOME": str(self.settings.home)},
            "StandardOutPath": str(self.settings.logs_path / "apriald.out.log"),
            "StandardErrorPath": str(self.settings.logs_path / "apriald.err.log"),
        }
        with path.open("wb") as fh:
            plistlib.dump(payload, fh, sort_keys=True)
        return path

    def uninstall(self) -> bool:
        path = self.plist_path
        self._validate_path(path)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed

    def status(self) -> dict[str, object]:
        path = self.plist_path
        self._validate_path(path)
        installed = path.exists()
        if self.platform != "darwin":
            return {
                "supported": False,
                "installed": installed,
                "loaded": False,
                "plist_path": str(path),
                "detail": "launchd is supported only on macOS",
            }
        result = self.runner(["launchctl", "print", self.service_target])
        return {
            "supported": True,
            "installed": installed,
            "loaded": result.returncode == 0,
            "plist_path": str(path),
        }

    @property
    def domain_target(self) -> str:
        return f"gui/{self.uid}"

    @property
    def service_target(self) -> str:
        return f"{self.domain_target}/{LABEL}"

    def bootstrap(self) -> dict[str, object]:
        if self.platform != "darwin":
            return self._unsupported()
        if bool(self.status()["loaded"]):
            return {"supported": True, "loaded": True, "changed": False}
        if not self.plist_path.exists():
            return {
                "supported": True,
                "loaded": False,
                "changed": False,
                "error": "LaunchAgent plist is not installed",
            }
        result = self.runner(["launchctl", "bootstrap", self.domain_target, str(self.plist_path)])
        return {
            "supported": True,
            "loaded": result.returncode == 0,
            "changed": result.returncode == 0,
            "returncode": result.returncode,
        }

    def bootout(self) -> dict[str, object]:
        if self.platform != "darwin":
            return self._unsupported()
        if not bool(self.status()["loaded"]):
            return {"supported": True, "loaded": False, "changed": False}
        result = self.runner(["launchctl", "bootout", self.service_target])
        return {
            "supported": True,
            "loaded": result.returncode != 0,
            "changed": result.returncode == 0,
            "returncode": result.returncode,
        }

    def kickstart(self) -> dict[str, object]:
        if self.platform != "darwin":
            return self._unsupported()
        if not bool(self.status()["loaded"]):
            return {
                "supported": True,
                "started": False,
                "error": "LaunchAgent is not loaded",
            }
        result = self.runner(["launchctl", "kickstart", "-k", self.service_target])
        return {
            "supported": True,
            "started": result.returncode == 0,
            "returncode": result.returncode,
        }

    def _unsupported(self) -> dict[str, object]:
        return {
            "supported": False,
            "changed": False,
            "detail": "launchd is supported only on macOS",
        }

    def _run(self, argv: Sequence[str]) -> RestrictedProcessResult:
        return run_restricted_process_sync(
            list(argv),
            cwd=self.user_home,
            category=ProcessCategory.DAEMON,
            timeout_seconds=30.0,
            max_stdout_bytes=20_000,
            max_stderr_bytes=20_000,
            april_home=self.settings.home,
        )

    def _validate_path(self, path: Path) -> None:
        expected_dir = self.launch_agents_dir.resolve(strict=False)
        parent = path.parent.resolve(strict=False)
        if parent != expected_dir:
            raise RuntimeError("launchd plist must be under the user's LaunchAgents directory.")
