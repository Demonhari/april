from __future__ import annotations

import asyncio
import json
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from april_common.audit import AuditLogger
from april_common.errors import RuntimeUnavailableError
from april_common.settings import AprilSettings


@dataclass(slots=True)
class Notification:
    """A single thing the scheduler wants to surface to the local user."""

    kind: str  # "reminder" | "briefing"
    title: str
    body: str
    reference_id: str | None = None
    created_at: str = ""

    def model_dump(self) -> dict[str, Any]:
        """Pydantic-compatible serialization so API handlers can treat this like a model."""
        return asdict(self)


class NotificationSink:
    """Pluggable delivery target, mirroring the SpeechToText/Fake* pattern."""

    async def emit(self, notification: Notification) -> None:
        raise NotImplementedError


class LogNotificationSink(NotificationSink):
    """Default, headless sink: records an audit entry and appends to scheduler.log."""

    def __init__(self, audit: AuditLogger, log_path: Path) -> None:
        self.audit = audit
        self.log_path = log_path

    async def emit(self, notification: Notification) -> None:
        self.audit.write(
            {
                "event": "scheduler.notification",
                "sink": "log",
                "kind": notification.kind,
                "title": notification.title,
                "reference_id": notification.reference_id,
            }
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "created_at": notification.created_at,
                "kind": notification.kind,
                "title": notification.title,
                "body": notification.body,
                "reference_id": notification.reference_id,
            },
            sort_keys=True,
        )
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class MacOsNotificationSink(NotificationSink):
    """Best-effort native banner via osascript. Guarded so it is never used in tests:
    it only runs on a real macOS host with osascript present, and raises otherwise."""

    def __init__(self, *, timeout: float = 10.0) -> None:
        self.timeout = timeout

    async def emit(self, notification: Notification) -> None:
        if platform.system() != "Darwin" or shutil.which("osascript") is None:
            raise RuntimeUnavailableError(
                "macOS notifications require a Darwin host with osascript available."
            )
        script = (
            f"display notification {_applescript_string(notification.body)} "
            f"with title {_applescript_string(notification.title)}"
        )
        process = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeUnavailableError("osascript notification timed out.") from exc
        if process.returncode:
            raise RuntimeUnavailableError(
                "osascript notification failed.",
                {"stderr": stderr.decode("utf-8", errors="replace")[:500]},
            )


class FakeNotificationSink(NotificationSink):
    """Records emitted notifications in memory for assertions."""

    def __init__(self) -> None:
        self.emitted: list[Notification] = []

    async def emit(self, notification: Notification) -> None:
        self.emitted.append(notification)


def _applescript_string(value: str) -> str:
    sanitized = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{sanitized}"'


def notification_sink_from_settings(
    settings: AprilSettings, audit: AuditLogger
) -> NotificationSink:
    if settings.scheduler.notification_sink == "macos":
        return MacOsNotificationSink()
    return LogNotificationSink(audit, settings.scheduler_log_path)
