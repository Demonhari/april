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
    remove_owned_socket,
    socket_identity,
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

    async def self_check(
        self,
        *,
        project_root: Path,
        timeout_seconds: float = 1.0,
    ) -> ToolWorkerResponse:
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await self.execute(
                    request_id=f"health-self-check:{secrets.token_hex(8)}",
                    operation="self_check",
                    project_root=project_root,
                    args={},
                    timeout_seconds=timeout_seconds,
                    max_stdout_bytes=0,
                    max_stderr_bytes=0,
                )
        except TimeoutError as exc:
            raise ToolWorkerUnavailable("tool_worker_self_check_timeout") from exc
        if not response.ok or response.data.get("self_check") is not True:
            raise ToolWorkerUnavailable("tool_worker_self_check_failed")
        return response

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
        environment: str = "development",
        development_unsandboxed_override: bool = False,
    ) -> None:
        self.april_home = april_home.expanduser().resolve(strict=True)
        self.allowed_roots = tuple(root.expanduser().resolve(strict=True) for root in allowed_roots)
        self._health_project_root = self.allowed_roots[0] if self.allowed_roots else self.april_home
        self.environment = environment
        self.development_unsandboxed_override = development_unsandboxed_override
        self.runtime_directory = runtime_directory or default_tool_worker_runtime_directory(
            self.april_home
        )
        self.socket_path = self.runtime_directory / "worker.sock"
        self.capability_path = self.runtime_directory / "capability"
        self.process: asyncio.subprocess.Process | None = None
        self._process_loop: asyncio.AbstractEventLoop | None = None
        self._owned_socket_identity: tuple[int, int] | None = None
        self._start_lock = asyncio.Lock()
        self.client = ToolWorkerClient(
            socket_path=self.socket_path,
            capability_path=self.capability_path,
            runtime_directory=self.runtime_directory,
        )

    async def start(self) -> ToolWorkerClient:
        async with self._start_lock:
            runtime = prepare_runtime_directory(
                self.runtime_directory,
                april_home=self.april_home,
            )
            if self.process is not None and self.process.returncode is not None:
                await self.stop()
            if os.path.lexists(self.socket_path):
                identity = socket_identity(self.socket_path, runtime_directory=runtime)
                # A socket is merely an endpoint. Authentication plus a bounded
                # health request is the only evidence that a worker is live.
                try:
                    read_capability_file(
                        self.capability_path,
                        runtime_directory=runtime,
                    )
                    await self.client.self_check(
                        project_root=self._health_project_root,
                        timeout_seconds=1.0,
                    )
                    return self.client
                except ToolWorkerUnavailable:
                    if os.environ.get("APRIL_TOOL_WORKER_EXTERNAL") == "1":
                        raise ToolWorkerUnavailable("tool_worker_external_unavailable") from None
                    # remove_owned_socket revalidates the parent, type, owner,
                    # mode, and inode. It cannot unlink a path that changed
                    # underneath this manager.
                    try:
                        remove_owned_socket(
                            self.socket_path,
                            runtime_directory=runtime,
                            identity=identity,
                        )
                    except UnsafeToolWorkerSocket as exc:
                        raise ToolWorkerUnavailable("unsafe_stale_worker_socket") from exc
                    if self.process is not None:
                        # The endpoint belonged to this manager but stopped
                        # answering. Do not leave the old process behind while
                        # replacing its validated stale endpoint.
                        await self.stop()
            elif os.environ.get("APRIL_TOOL_WORKER_EXTERNAL") == "1":
                raise ToolWorkerUnavailable("tool_worker_external_unavailable")
            elif self.process is not None:
                # A live child without its endpoint is not a reusable worker.
                await self.stop()

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
                "--environment",
                self.environment,
            ]
            if self.development_unsandboxed_override:
                if self.environment != "development":
                    raise ToolWorkerUnavailable("unsandboxed_override_development_only")
                argv.append("--development-unsandboxed-override")
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
            self._process_loop = asyncio.get_running_loop()
            for _ in range(50):
                process = self.process
                if process is None or process.returncode is not None:
                    await self.stop()
                    raise ToolWorkerUnavailable("tool_worker_start_failed")
                if os.path.lexists(self.socket_path):
                    validate_live_socket(self.socket_path, runtime_directory=runtime)
                    self._owned_socket_identity = socket_identity(
                        self.socket_path,
                        runtime_directory=runtime,
                    )
                    try:
                        await self.client.self_check(
                            project_root=self._health_project_root,
                            timeout_seconds=1.0,
                        )
                    except ToolWorkerUnavailable:
                        await self.stop()
                        raise ToolWorkerUnavailable("tool_worker_self_check_failed") from None
                    return self.client
                await asyncio.sleep(0.02)
            await self.stop()
            raise ToolWorkerUnavailable("tool_worker_start_timeout")

    async def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                if self._process_loop is asyncio.get_running_loop():
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except TimeoutError:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()
                else:
                    await _reap_process_from_other_loop(process)
            else:
                # A known return code does not replace an explicit wait: always
                # reap the asyncio child handle before declaring shutdown done.
                if self._process_loop is asyncio.get_running_loop():
                    await process.wait()
            _close_process_transport(process)
            self._process_loop = None
        identity = self._owned_socket_identity
        self._owned_socket_identity = None
        if identity is not None:
            with suppress(FileNotFoundError, UnsafeToolWorkerSocket):
                runtime = prepare_runtime_directory(
                    self.runtime_directory,
                    april_home=self.april_home,
                )
                remove_owned_socket(
                    self.socket_path,
                    runtime_directory=runtime,
                    identity=identity,
                )


async def _reap_process_from_other_loop(process: asyncio.subprocess.Process) -> None:
    """Terminate/reap a child whose asyncio Process belongs to another loop."""
    deadline = asyncio.get_running_loop().time() + 2.0
    while asyncio.get_running_loop().time() < deadline:
        try:
            child_pid, _status = os.waitpid(process.pid, os.WNOHANG)
        except ChildProcessError:
            _close_process_transport(process)
            return
        if child_pid == process.pid:
            _close_process_transport(process)
            return
        await asyncio.sleep(0.02)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with suppress(ChildProcessError):
        await asyncio.to_thread(os.waitpid, process.pid, 0)
    _close_process_transport(process)


def _close_process_transport(process: asyncio.subprocess.Process) -> None:
    """Close an asyncio child transport after cross-loop OS-level reaping."""
    transport = getattr(process, "_transport", None)
    if transport is not None:
        with suppress(Exception):
            transport.close()
