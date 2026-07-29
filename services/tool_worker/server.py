from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
from collections import OrderedDict
from pathlib import Path

from services.tool_worker.executor import ToolWorkerExecutor
from services.tool_worker.limits import (
    UnsafeToolWorkerSocket,
    prepare_runtime_directory,
    prepare_socket_path,
    read_capability_file,
)
from services.tool_worker.protocol import (
    ToolWorkerProtocolError,
    message_fingerprint,
    read_request,
    safe_error_response,
    write_message,
)
from services.tool_worker.schemas import (
    MAX_TOOL_WORKER_RESPONSE_BYTES,
    ToolWorkerResponse,
)

_IDEMPOTENCY_RETENTION = 256
_IDEMPOTENCY_JOURNAL_VERSION = 1
_MAX_IDEMPOTENCY_JOURNAL_BYTES = 128 * 1024


class ToolWorkerServer:
    def __init__(
        self,
        *,
        april_home: Path,
        socket_path: Path,
        capability_path: Path,
        allowed_roots: tuple[Path, ...],
        environment: str | None = None,
        development_unsandboxed_override: bool = False,
    ) -> None:
        self.april_home = april_home
        self.runtime_directory = socket_path.parent
        self.socket_path = socket_path
        self.capability_path = capability_path
        self.allowed_roots = allowed_roots
        self.environment = environment
        self.development_unsandboxed_override = development_unsandboxed_override
        self._server: asyncio.Server | None = None
        self._outcomes: OrderedDict[str, tuple[str, ToolWorkerResponse | None, bool]] = (
            OrderedDict()
        )
        self._journal_path = self.runtime_directory / "request-outcomes.json"

    async def start(self) -> None:
        runtime = prepare_runtime_directory(
            self.runtime_directory,
            april_home=self.april_home,
        )
        self._load_outcomes(runtime)
        socket_path = prepare_socket_path(self.socket_path, runtime_directory=runtime)
        capability = read_capability_file(
            self.capability_path,
            runtime_directory=runtime,
        )
        self.executor = ToolWorkerExecutor(
            allowed_roots=self.allowed_roots,
            capability=capability,
            environment=self.environment,
            development_unsandboxed_override=self.development_unsandboxed_override,
        )
        self._server = await asyncio.start_unix_server(self._handle, path=socket_path)
        os.chmod(socket_path, 0o600)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self.socket_path.exists() and not self.socket_path.is_symlink():
            self.socket_path.unlink()

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_id = "unknown"
        try:
            request = await read_request(reader)
            request_id = request.request_id
            fingerprint = message_fingerprint(
                request.model_dump(mode="json", exclude={"capability"})
            )
            retained = self._outcomes.get(request_id)
            if retained is not None:
                prior_fingerprint, prior_response, completed = retained
                response = (
                    prior_response
                    if prior_response is not None
                    else safe_error_response(
                        request_id,
                        "duplicate_request_completed"
                        if completed
                        else "duplicate_request_interrupted",
                    )
                    if hmac_compare(fingerprint, prior_fingerprint)
                    else safe_error_response(request_id, "duplicate_request_mismatch")
                )
            else:
                # Persist intent before execution. A crash after an external
                # mutation but before its outcome is stored therefore fails a
                # duplicate closed instead of replaying it.
                self._outcomes[request_id] = (fingerprint, None, False)
                self._prune_and_persist_outcomes()
                response = await self.executor.execute(request)
                self._outcomes[request_id] = (fingerprint, response, True)
                self._prune_and_persist_outcomes()
        except ToolWorkerProtocolError as exc:
            response = safe_error_response(request_id, str(exc))
        except Exception:
            response = safe_error_response(request_id, "worker_internal_error")
        try:
            await write_message(
                writer,
                response,
                maximum=MAX_TOOL_WORKER_RESPONSE_BYTES,
            )
        except (OSError, ToolWorkerProtocolError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    def _load_outcomes(self, runtime_directory: Path) -> None:
        path = self._journal_path
        if path.parent.resolve(strict=True) != runtime_directory.resolve(strict=True):
            raise UnsafeToolWorkerSocket("idempotency_journal_outside_runtime_directory")
        if not path.exists():
            return
        if path.is_symlink():
            raise UnsafeToolWorkerSocket("idempotency_journal_is_symlink")
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise UnsafeToolWorkerSocket("unsafe_idempotency_journal")
        if info.st_size > _MAX_IDEMPOTENCY_JOURNAL_BYTES:
            raise UnsafeToolWorkerSocket("idempotency_journal_too_large")
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
            if decoded.get("version") != _IDEMPOTENCY_JOURNAL_VERSION:
                raise ValueError
            entries = decoded.get("entries")
            if not isinstance(entries, list) or len(entries) > _IDEMPOTENCY_RETENTION:
                raise ValueError
            loaded: OrderedDict[str, tuple[str, ToolWorkerResponse | None, bool]] = OrderedDict()
            for entry in entries:
                request_id = str(entry["request_id"])
                fingerprint = str(entry["fingerprint"])
                completed = bool(entry["completed"])
                if (
                    not 1 <= len(request_id) <= 128
                    or len(fingerprint) != 64
                    or any(char not in "0123456789abcdef" for char in fingerprint)
                ):
                    raise ValueError
                loaded[request_id] = (fingerprint, None, completed)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise UnsafeToolWorkerSocket("invalid_idempotency_journal") from exc
        self._outcomes = loaded

    def _prune_and_persist_outcomes(self) -> None:
        while len(self._outcomes) > _IDEMPOTENCY_RETENTION:
            self._outcomes.popitem(last=False)
        payload = {
            "version": _IDEMPOTENCY_JOURNAL_VERSION,
            "entries": [
                {
                    "request_id": request_id,
                    "fingerprint": fingerprint,
                    "completed": completed,
                }
                for request_id, (fingerprint, _response, completed) in self._outcomes.items()
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > _MAX_IDEMPOTENCY_JOURNAL_BYTES:
            raise UnsafeToolWorkerSocket("idempotency_journal_too_large")
        temporary = self._journal_path.with_suffix(".tmp")
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, self._journal_path)


def hmac_compare(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left, right)


async def _run(args: argparse.Namespace) -> None:
    roots = tuple(Path(value).expanduser().resolve(strict=True) for value in args.allowed_root)
    server = ToolWorkerServer(
        april_home=Path(args.april_home).expanduser().resolve(strict=True),
        socket_path=Path(args.socket),
        capability_path=Path(args.capability_file),
        allowed_roots=roots,
        environment=args.environment,
        development_unsandboxed_override=args.development_unsandboxed_override,
    )
    try:
        await server.serve_forever()
    finally:
        await server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="APRIL owner-only local Tool Worker")
    parser.add_argument("--april-home", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--capability-file", required=True)
    parser.add_argument("--allowed-root", action="append", required=True)
    parser.add_argument(
        "--environment",
        choices=("development", "test", "production"),
        default="development",
    )
    parser.add_argument(
        "--development-unsandboxed-override",
        action="store_true",
        help="DEVELOPMENT ONLY: permit subprocesses without an OS sandbox.",
    )
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
