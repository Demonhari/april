from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from april_common.settings import AprilSettings


def sentinel_control_path(settings: AprilSettings) -> Path:
    return settings.resolve_path(Path("data/sentinel-control.sock"))


class SentinelControlServer:
    """Owner-only IPC used to attach a terminal to the resident Sentinel."""

    def __init__(
        self,
        path: Path,
        *,
        set_session_hint: Callable[[str | None], None],
        status: Callable[[], dict[str, Any]],
    ) -> None:
        self.path = path
        self.set_session_hint = set_session_hint
        self.status = status
        self._server: asyncio.AbstractServer | None = None
        self._controller_id: str | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if await _socket_responds(self.path):
                raise RuntimeError("A resident Sentinel already owns the control socket.")
            self.path.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(self._handle, path=str(self.path))
        os.chmod(self.path, 0o600)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.set_session_hint(None)
        self._controller_id = None
        self.path.unlink(missing_ok=True)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        attached_id: str | None = None
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=2.0)
            request = json.loads(raw.decode("utf-8"))
            action = request.get("action") if isinstance(request, dict) else None
            if action == "status":
                await _reply(
                    writer,
                    {
                        "ok": True,
                        "controlled": self._controller_id is not None,
                        **self.status(),
                    },
                )
                return
            if action != "attach":
                await _reply(writer, {"ok": False, "error": "unsupported control action"})
                return
            controller_id = str(request.get("controller_id") or "")
            session_hint = str(request.get("session_hint") or "")
            if not controller_id or not session_hint:
                await _reply(writer, {"ok": False, "error": "missing attachment identity"})
                return
            async with self._lock:
                if self._controller_id is not None:
                    await _reply(
                        writer,
                        {"ok": False, "error": "Sentinel is already controlled elsewhere."},
                    )
                    return
                self._controller_id = controller_id
                attached_id = controller_id
                self.set_session_hint(session_hint)
            await _reply(writer, {"ok": True, "attached": True, **self.status()})
            # The live connection is the lease. EOF/Ctrl-C/process death
            # releases control without a stale owner record.
            await reader.read()
        except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
            await _reply(writer, {"ok": False, "error": "malformed control request"})
        finally:
            if attached_id is not None:
                async with self._lock:
                    if self._controller_id == attached_id:
                        self._controller_id = None
                        self.set_session_hint(None)
            writer.close()
            await writer.wait_closed()


@dataclass(slots=True)
class ResidentSentinelAttachment:
    sock: socket.socket
    status: dict[str, Any]

    def close(self) -> None:
        self.sock.close()

    def __enter__(self) -> ResidentSentinelAttachment:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def attach_resident_sentinel(
    settings: AprilSettings,
    *,
    session_hint: str,
    timeout_seconds: float = 2.0,
) -> ResidentSentinelAttachment:
    path = sentinel_control_path(settings)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_seconds)
    try:
        sock.connect(str(path))
        request = {
            "action": "attach",
            "controller_id": str(uuid.uuid4()),
            "session_hint": session_hint,
        }
        sock.sendall(json.dumps(request, sort_keys=True).encode("utf-8") + b"\n")
        response = _recv_line(sock)
        payload = json.loads(response.decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError(
                str(payload.get("error", "resident Sentinel refused attachment"))
                if isinstance(payload, dict)
                else "invalid resident Sentinel response"
            )
        sock.settimeout(None)
        return ResidentSentinelAttachment(sock=sock, status=payload)
    except Exception:
        sock.close()
        raise


def resident_sentinel_status(
    settings: AprilSettings, *, timeout_seconds: float = 1.0
) -> dict[str, Any]:
    """Read truthful resident-Sentinel state without acquiring its control lease."""
    return sentinel_status_at_path(sentinel_control_path(settings), timeout_seconds=timeout_seconds)


def sentinel_status_at_path(path: Path, *, timeout_seconds: float = 1.0) -> dict[str, Any]:
    """Read status from an explicitly scoped Sentinel control socket."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_seconds)
    try:
        sock.connect(str(path))
        sock.sendall(b'{"action":"status"}\n')
        payload = json.loads(_recv_line(sock).decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError("resident Sentinel returned invalid status")
        return payload
    finally:
        sock.close()


async def _socket_responds(path: Path) -> bool:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(path)), timeout=0.5
        )
        writer.write(b'{"action":"status"}\n')
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=0.5)
        writer.close()
        await writer.wait_closed()
        payload = json.loads(raw.decode("utf-8"))
        return isinstance(payload, dict) and payload.get("ok") is True
    except (OSError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
        return False


async def _reply(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    writer.write(json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n")
    await writer.drain()


def _recv_line(sock: socket.socket, *, max_bytes: int = 8192) -> bytes:
    chunks = bytearray()
    while len(chunks) < max_bytes:
        chunk = sock.recv(1)
        if not chunk:
            break
        chunks.extend(chunk)
        if chunk == b"\n":
            return bytes(chunks)
    raise RuntimeError("resident Sentinel returned an incomplete response")
