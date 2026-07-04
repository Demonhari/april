from __future__ import annotations

import plistlib
import sys
from pathlib import Path

from april_common.settings import AprilSettings

LABEL = "com.april.apriald"


class LaunchdManager:
    def __init__(self, settings: AprilSettings, *, user_home: Path | None = None) -> None:
        self.settings = settings
        self.user_home = (user_home or Path.home()).expanduser().resolve()

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
        return {"installed": path.exists(), "plist_path": str(path)}

    def _validate_path(self, path: Path) -> None:
        expected_dir = self.launch_agents_dir.resolve(strict=False)
        parent = path.parent.resolve(strict=False)
        if parent != expected_dir:
            raise RuntimeError("launchd plist must be under the user's LaunchAgents directory.")
