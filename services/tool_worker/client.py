from __future__ import annotations

import asyncio
import os
import secrets
import signal
import sys
from contextlib import suppress
from pathlib import Path

from april_common.process_environment import ProcessCategory, build_process_environment
from services.tool_worker.limits import (
    UnsafeToolWorkerSocket,
    default_tool_worker_runtime_directory,
    prepare_runtime_directory,
    read_capability_file,
    validate_live_socket,
    write_capability_file,
)
from services.tool_worker.protocol import (
    ToolWorkerProtocolError,
    read_response,
    write_message,
)
from services.tool_worker.schemas import (
    MAX_TOOL_WORKER_OUTPUT_BYTES,
    MAX_TOOL_WORKER_REQUEST_BYTES,
    ToolWorkerRequest,
    ToolWorkerResponse,
)


class ToolWorkerUnavailable(RuntimeError):
    pass


class ToolWorkerClient:
    def __init__(
        self,
        *,
        socket_path: Path,
        capability_path: Path,
        runtime_directory: Path,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        self.socket_path = socket_path
        self.capability_path = capability_path
        self.runtime_directory = runtime_directory
        self.request_timeout_seconds = request_timeout_seconds

    async def execute(
        self,
        *,
        request_id: str,
        operation: str,
        project_root: Path,
        args: dict[str, object],
        timeout_seconds: float,
        max_stdout_bytes: int = MAX_TOOL_WORKER_OUTPUT_BYTES,
        max_stderr_bytes: int = MAX_TOOL_WORKER_OUTPUT_BYTES,
    ) -> ToolWorkerResponse:
        try:
            validate_live_socket(
                self.socket_path,
                runtime_directory=self.runtime_directory,
            )
            capability = read_capability_file(
                self.capability_path,
                runtime_directory=self.runtime_directory,
            )
            request = ToolWorkerRequest(
                request_id=request_id,
                capability=capability,
                operation=operation,
                project_root=str(project_root),
                args=args,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            )
            async with asyncio.timeout(self.request_timeout_seconds + timeout_seconds):
                reader, writer = await asyncio.open_unix_connection(self.socket_path)
                try:
                    await write_message(
                        writer,
                        request,
                        maximum=MAX_TOOL_WORKER_REQUEST_BYTES,
                    )
                    response = await read_response(reader)
                finally:
                    writer.close()
                    await writer.wait_closed()
        except (
            OSError,
            TimeoutError,
            UnsafeToolWorkerSocket,
            ToolWorkerProtocolError,
        ) as exc:
            raise ToolWorkerUnavailable("tool_worker_unavailable") from exc
        if response.request_id != request_id:
            raise ToolWorkerUnavailable("tool_worker_response_mismatch")
        return response

    async def self_check(self, *, project_root: Path) -> ToolWorkerResponse:
        return await self.execute(
            request_id=f"health-self-check:{secrets.token_hex(8)}",
            operation="self_check",
            project_root=project_root,
            args={},
            timeout_seconds=5.0,
            max_stdout_bytes=0,
            max_stderr_bytes=0,
        )

    async def cancel(self, *, target_request_id: str, project_root: Path) -> bool:
        response = await self.execute(
            request_id=f"cancel:{secrets.token_hex(8)}",
            operation="cancel",
            project_root=project_root,
            args={"target_request_id": target_request_id},
            timeout_seconds=5.0,
            max_stdout_bytes=0,
            max_stderr_bytes=0,
        )
        return bool(response.data.get("cancellation_signalled"))


class ToolWorkerProcessManager:
    """Start a Tool Worker only when one is not externally supervised."""

    def __init__(
        self,
        *,
        april_home: Path,
        allowed_roots: tuple[Path, ...],
        runtime_directory: Path | None = None,
    ) -> None:
        self.april_home = april_home.expanduser().resolve(strict=True)
        self.allowed_roots = allowed_roots
        self.runtime_directory = runtime_directory or default_tool_worker_runtime_directory(
            self.april_home
        )
        self.socket_path = self.runtime_directory / "worker.sock"
        self.capability_path = self.runtime_directory / "capability"
        self.process: asyncio.subprocess.Process | None = None
        self.client = ToolWorkerClient(
            socket_path=self.socket_path,
            capability_path=self.capability_path,
            runtime_directory=self.runtime_directory,
        )

    async def start(self) -> ToolWorkerClient:
        runtime = prepare_runtime_directory(
            self.runtime_directory,
            april_home=self.april_home,
        )
        if self.socket_path.exists():
            validate_live_socket(self.socket_path, runtime_directory=runtime)
            return self.client
        if os.environ.get("APRIL_TOOL_WORKER_EXTERNAL") == "1":
            raise ToolWorkerUnavailable("tool_worker_not_running")
        capability = secrets.token_urlsafe(32)
        write_capability_file(
            self.capability_path,
            capability,
            runtime_directory=runtime,
        )
        argv = [
            sys.executable,
            "-m",
            "services.tool_worker.server",
            "--april-home",
            str(self.april_home),
            "--socket",
            str(self.socket_path),
            "--capability-file",
            str(self.capability_path),
        ]
        for root in self.allowed_roots:
            argv.extend(["--allowed-root", str(root)])
        self.process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self.april_home),
            env=build_process_environment(
                ProcessCategory.TOOL_WORKER,
                april_home=self.april_home,
            ),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(50):
            if self.process.returncode is not None:
                raise ToolWorkerUnavailable("tool_worker_start_failed")
            if self.socket_path.exists():
                validate_live_socket(self.socket_path, runtime_directory=runtime)
                return self.client
            await asyncio.sleep(0.02)
        await self.stop()
        raise ToolWorkerUnavailable("tool_worker_start_timeout")

    async def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
