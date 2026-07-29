from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

from pydantic import ValidationError

from services.tool_worker.schemas import (
    MAX_TOOL_WORKER_REQUEST_BYTES,
    MAX_TOOL_WORKER_RESPONSE_BYTES,
    ToolWorkerRequest,
    ToolWorkerResponse,
)

_HEADER = struct.Struct("!I")


class ToolWorkerProtocolError(RuntimeError):
    pass


async def read_request(reader: asyncio.StreamReader) -> ToolWorkerRequest:
    payload = await _read_frame(reader, MAX_TOOL_WORKER_REQUEST_BYTES)
    try:
        decoded = json.loads(payload)
        return ToolWorkerRequest.model_validate(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ToolWorkerProtocolError("malformed_request") from exc


async def read_response(reader: asyncio.StreamReader) -> ToolWorkerResponse:
    payload = await _read_frame(reader, MAX_TOOL_WORKER_RESPONSE_BYTES)
    try:
        decoded = json.loads(payload)
        return ToolWorkerResponse.model_validate(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ToolWorkerProtocolError("malformed_response") from exc


async def write_message(
    writer: asyncio.StreamWriter,
    message: ToolWorkerRequest | ToolWorkerResponse,
    *,
    maximum: int,
) -> None:
    payload = json.dumps(
        message.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(payload) > maximum:
        raise ToolWorkerProtocolError("message_too_large")
    writer.write(_HEADER.pack(len(payload)))
    writer.write(payload)
    await writer.drain()


async def _read_frame(reader: asyncio.StreamReader, maximum: int) -> str:
    try:
        header = await reader.readexactly(_HEADER.size)
    except asyncio.IncompleteReadError as exc:
        raise ToolWorkerProtocolError("incomplete_header") from exc
    (size,) = _HEADER.unpack(header)
    if size <= 0 or size > maximum:
        raise ToolWorkerProtocolError("message_too_large")
    try:
        payload = await reader.readexactly(size)
    except asyncio.IncompleteReadError as exc:
        raise ToolWorkerProtocolError("incomplete_message") from exc
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolWorkerProtocolError("message_not_utf8") from exc


def safe_error_response(request_id: str, code: str) -> ToolWorkerResponse:
    return ToolWorkerResponse(
        request_id=request_id[:128] or "unknown",
        ok=False,
        status="rejected",
        failure_code=code[:160],
    )


def message_fingerprint(value: dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
