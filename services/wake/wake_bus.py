from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import ValidationError

from services.wake.schemas import WakeEvent

logger = logging.getLogger(__name__)

WakeHandler = Callable[[WakeEvent], Awaitable[dict[str, object] | None]]

_MAX_LINE_BYTES = 65_536


class WakeBus:
    """Loopback-free local wake bus over a Unix domain socket.

    Local processes (hotkey helper, desktop shell, scripts) deliver JSON
    ``WakeEvent`` lines to ``data/wake.sock``. The socket is created with mode
    0600 so only the owning user can connect. Startup removes a stale socket
    file, shutdown closes the server and unlinks the path.
    """

    def __init__(self, path: Path, handler: WakeHandler) -> None:
        self.path = path
        self.handler = handler
        self._server: asyncio.AbstractServer | None = None

    @property
    def running(self) -> bool:
        return self._server is not None

    async def start(self) -> None:
        if self._server is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_socket()
        # Pre-restrict via umask so the socket is never observable with wider
        # permissions, then enforce 0600 explicitly.
        previous_umask = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_connection, path=str(self.path)
            )
        finally:
            os.umask(previous_umask)
        os.chmod(self.path, 0o600)

    async def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)

    def _remove_stale_socket(self) -> None:
        try:
            mode = os.lstat(self.path).st_mode
        except FileNotFoundError:
            return
        # Never delete a non-socket path: a regular file/dir/symlink here is
        # unexpected and refusing is safer than clobbering user data.
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(f"Wake bus path exists and is not a socket: {self.path}")
        self.path.unlink(missing_ok=True)

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                try:
                    line = await reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    await self._reply(writer, {"ok": False, "error": "line too long"})
                    break
                if not line:
                    break
                if len(line) > _MAX_LINE_BYTES:
                    await self._reply(writer, {"ok": False, "error": "line too long"})
                    continue
                response = await self._handle_line(line)
                await self._reply(writer, response)
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _handle_line(self, line: bytes) -> dict[str, object]:
        try:
            payload = json.loads(line.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return {"ok": False, "error": "invalid json"}
        try:
            event = WakeEvent.model_validate(payload)
        except ValidationError:
            return {"ok": False, "error": "invalid wake event"}
        try:
            result = await self.handler(event)
        except Exception as exc:  # keep the bus alive across handler failures
            logger.warning("Wake bus handler failed: %s", exc)
            return {"ok": False, "error": "wake handler failed"}
        return {"ok": True, "result": result or {}}

    async def _reply(self, writer: asyncio.StreamWriter, payload: dict[str, object]) -> None:
        with contextlib.suppress(Exception):
            writer.write(json.dumps(payload).encode("utf-8") + b"\n")
            await writer.drain()


async def send_wake_event(
    path: Path, event: WakeEvent, *, timeout: float = 5.0
) -> dict[str, object]:
    """Deliver one wake event to a running wake bus and return its reply."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path=str(path)), timeout=timeout
    )
    try:
        writer.write(event.model_dump_json().encode("utf-8") + b"\n")
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid bus reply"}
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": "invalid bus reply"}
