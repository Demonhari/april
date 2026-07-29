from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from april_common.process_environment import ProcessCategory, build_process_environment
from april_common.process_sandbox import (
    HostProcessSandbox,
    SandboxCapabilities,
    SandboxPolicy,
    SandboxProvider,
    SandboxUnavailableError,
)

DEFAULT_MAX_OUTPUT_BYTES = 100_000
DEFAULT_TERMINATION_GRACE_SECONDS = 1.0
MAX_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_OUTPUT_BYTES = 10_000_000


class ProcessStatus(StrEnum):
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    START_FAILED = "start_failed"


class ResourceLimitProfile(StrEnum):
    NONE = "none"
    COMMAND = "command"
    TEST = "test"
    PATCH = "patch"
    INDEXING = "indexing"
    MODEL_UTILITY = "model_utility"
    TRAINING = "training"


@dataclass(frozen=True, slots=True)
class ResourceLimitReport:
    requested_profile: ResourceLimitProfile
    applied: tuple[str, ...]
    unsupported: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RestrictedProcessResult:
    status: ProcessStatus
    returncode: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_seconds: float
    failure_code: str | None
    resource_limits: ResourceLimitReport
    sandbox: SandboxCapabilities | None = None


AsyncProcessLauncher = Callable[..., Awaitable[asyncio.subprocess.Process]]


@dataclass(frozen=True, slots=True)
class _LimitValues:
    cpu_seconds: int | None = None
    address_space_bytes: int | None = None
    open_files: int | None = None
    process_count: int | None = None
    file_size_bytes: int | None = None


_LIMITS: dict[ResourceLimitProfile, _LimitValues] = {
    ResourceLimitProfile.NONE: _LimitValues(),
    ResourceLimitProfile.COMMAND: _LimitValues(300, 2 * 1024**3, 256, 64, 32 * 1024**2),
    ResourceLimitProfile.TEST: _LimitValues(1800, 4 * 1024**3, 512, 128, 64 * 1024**2),
    ResourceLimitProfile.PATCH: _LimitValues(60, 1024**3, 128, 32, 16 * 1024**2),
    ResourceLimitProfile.INDEXING: _LimitValues(1800, 4 * 1024**3, 512, 64, 64 * 1024**2),
    ResourceLimitProfile.MODEL_UTILITY: _LimitValues(3600, 16 * 1024**3, 512, 64, 128 * 1024**2),
    ResourceLimitProfile.TRAINING: _LimitValues(
        14_400,
        8 * 1024**3,
        512,
        64,
        1024 * 1024**2,
    ),
}


async def run_restricted_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    category: ProcessCategory,
    timeout_seconds: float,
    max_stdout_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_stderr_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    cancellation_event: asyncio.Event | None = None,
    resource_limit_profile: ResourceLimitProfile = ResourceLimitProfile.NONE,
    stdin_bytes: bytes | None = None,
    april_home: Path | None = None,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    sandbox_policy: SandboxPolicy | None = None,
    sandbox_environment: str = "development",
    development_unsandboxed_override: bool = False,
    sandbox_provider: SandboxProvider | None = None,
    process_launcher: AsyncProcessLauncher | None = None,
) -> RestrictedProcessResult:
    """Execute one argv-only child in an isolated process group with hard bounds."""
    normalized_argv = _validate_argv(argv)
    resolved_cwd = cwd.expanduser().resolve(strict=True)
    if not resolved_cwd.is_dir():
        raise ValueError("Process cwd must be an existing directory.")
    timeout = _bounded_float(timeout_seconds, 0.01, MAX_TIMEOUT_SECONDS, "timeout")
    stdout_limit = _bounded_int(max_stdout_bytes, 0, MAX_OUTPUT_BYTES, "stdout limit")
    stderr_limit = _bounded_int(max_stderr_bytes, 0, MAX_OUTPUT_BYTES, "stderr limit")
    grace = _bounded_float(termination_grace_seconds, 0.01, 30.0, "termination grace")
    environment = build_process_environment(category, april_home=april_home)
    limit_report, preexec_fn = _resource_limit_setup(resource_limit_profile)
    started = time.monotonic()
    sandbox_capabilities: SandboxCapabilities | None = None
    process: asyncio.subprocess.Process | None = None
    stdout_task: asyncio.Task[tuple[bytes, bool]] | None = None
    stderr_task: asyncio.Task[tuple[bytes, bool]] | None = None
    stdin_task: asyncio.Task[None] | None = None
    try:
        launch_argv = normalized_argv
        if sandbox_policy is not None:
            launch = (sandbox_provider or HostProcessSandbox()).wrap(
                normalized_argv,
                policy=sandbox_policy,
                environment=sandbox_environment,
                development_override=development_unsandboxed_override,
            )
            launch_argv = launch.argv
            sandbox_capabilities = launch.capabilities
        launcher = process_launcher or asyncio.create_subprocess_exec
        process = await launcher(
            *launch_argv,
            cwd=str(resolved_cwd),
            env=environment,
            stdin=asyncio.subprocess.PIPE
            if stdin_bytes is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            preexec_fn=preexec_fn,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_task = asyncio.create_task(_bounded_read(process.stdout, stdout_limit))
        stderr_task = asyncio.create_task(_bounded_read(process.stderr, stderr_limit))
        if stdin_bytes is not None:
            assert process.stdin is not None
            stdin_task = asyncio.create_task(_write_stdin(process.stdin, stdin_bytes))

        wait_task = asyncio.create_task(process.wait())
        cancel_task = (
            asyncio.create_task(cancellation_event.wait())
            if cancellation_event is not None
            else None
        )
        watched: set[asyncio.Task[Any]] = {wait_task}
        if cancel_task is not None:
            watched.add(cancel_task)
        done, _pending = await asyncio.wait(
            watched,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        status = ProcessStatus.COMPLETED
        failure_code: str | None = None
        if wait_task not in done:
            if (
                cancel_task is not None
                and cancel_task in done
                and cancellation_event is not None
                and cancellation_event.is_set()
            ):
                status = ProcessStatus.CANCELLED
                failure_code = "cancelled"
            else:
                status = ProcessStatus.TIMED_OUT
                failure_code = "timeout"
            await _terminate_process_group(process, grace)
            await wait_task
        if cancel_task is not None:
            cancel_task.cancel()
        if stdin_task is not None:
            await _quiet_task(stdin_task)
        stdout_bytes, stdout_truncated = await stdout_task
        stderr_bytes, stderr_truncated = await stderr_task
        return RestrictedProcessResult(
            status=status,
            returncode=process.returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            duration_seconds=max(0.0, time.monotonic() - started),
            failure_code=failure_code,
            resource_limits=limit_report,
            sandbox=sandbox_capabilities,
        )
    except asyncio.CancelledError:
        if process is not None:
            await asyncio.shield(_terminate_process_group(process, grace))
        raise
    except (OSError, ValueError, SandboxUnavailableError) as exc:
        if process is not None and process.returncode is None:
            await _terminate_process_group(process, grace)
        return RestrictedProcessResult(
            status=ProcessStatus.START_FAILED,
            returncode=None,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=max(0.0, time.monotonic() - started),
            failure_code=_safe_start_failure_code(exc),
            resource_limits=limit_report,
            sandbox=sandbox_capabilities,
        )
    finally:
        for task in (stdin_task, stdout_task, stderr_task):
            if task is not None and not task.done():
                task.cancel()
                await _quiet_task(task)


def run_restricted_process_sync(
    argv: Sequence[str],
    *,
    cwd: Path,
    category: ProcessCategory,
    timeout_seconds: float,
    max_stdout_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_stderr_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    resource_limit_profile: ResourceLimitProfile = ResourceLimitProfile.NONE,
    april_home: Path | None = None,
    sandbox_policy: SandboxPolicy | None = None,
    sandbox_environment: str = "development",
    development_unsandboxed_override: bool = False,
    sandbox_provider: SandboxProvider | None = None,
    process_launcher: AsyncProcessLauncher | None = None,
) -> RestrictedProcessResult:
    """Synchronous adapter for startup and diagnostic paths without an event loop."""
    return asyncio.run(
        run_restricted_process(
            argv,
            cwd=cwd,
            category=category,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            resource_limit_profile=resource_limit_profile,
            april_home=april_home,
            sandbox_policy=sandbox_policy,
            sandbox_environment=sandbox_environment,
            development_unsandboxed_override=development_unsandboxed_override,
            sandbox_provider=sandbox_provider,
            process_launcher=process_launcher,
        )
    )


async def _bounded_read(
    stream: asyncio.StreamReader,
    maximum: int,
) -> tuple[bytes, bool]:
    retained = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(16_384)
        if not chunk:
            break
        remaining = maximum - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > max(0, remaining):
            truncated = True
    return bytes(retained), truncated


async def _write_stdin(writer: asyncio.StreamWriter, value: bytes) -> None:
    try:
        writer.write(value)
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        writer.close()


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
    grace_seconds: float,
) -> None:
    if process.returncode is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return
    except TimeoutError:
        pass
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


def _resource_limit_setup(
    profile: ResourceLimitProfile,
) -> tuple[ResourceLimitReport, Any]:
    values = _LIMITS[profile]
    if profile == ResourceLimitProfile.NONE:
        return ResourceLimitReport(profile, (), ()), None
    if os.name != "posix":
        names = ("cpu_time", "address_space", "open_files", "process_count", "file_size")
        return ResourceLimitReport(profile, (), names), None
    try:
        import resource
    except ImportError:
        names = ("cpu_time", "address_space", "open_files", "process_count", "file_size")
        return ResourceLimitReport(profile, (), names), None

    settings: list[tuple[str, int, int]] = []
    unsupported: list[str] = []
    candidates = (
        ("cpu_time", "RLIMIT_CPU", values.cpu_seconds),
        ("address_space", "RLIMIT_AS", values.address_space_bytes),
        ("open_files", "RLIMIT_NOFILE", values.open_files),
        ("process_count", "RLIMIT_NPROC", values.process_count),
        ("file_size", "RLIMIT_FSIZE", values.file_size_bytes),
    )
    for name, symbol, value in candidates:
        if value is None:
            continue
        limit_id = getattr(resource, symbol, None)
        if limit_id is None or (name == "address_space" and sys.platform == "darwin"):
            unsupported.append(name)
        else:
            settings.append((name, limit_id, value))

    def apply_limits() -> None:
        for _name, limit_id, requested in settings:
            _soft, hard = resource.getrlimit(limit_id)
            effective = requested if hard < 0 else min(requested, hard)
            resource.setrlimit(limit_id, (effective, effective))

    return (
        ResourceLimitReport(
            profile,
            tuple(name for name, _limit, _value in settings),
            tuple(unsupported),
        ),
        apply_limits,
    )


def resource_limit_report(profile: ResourceLimitProfile) -> ResourceLimitReport:
    """Return truthful platform support without changing the current process."""
    report, _apply = _resource_limit_setup(profile)
    return report


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("argv must be a non-empty sequence of strings.")
    normalized: list[str] = []
    for item in argv:
        if not isinstance(item, str) or not item or "\x00" in item:
            raise ValueError("argv items must be non-empty strings without NUL bytes.")
        normalized.append(item)
    return tuple(normalized)


def _bounded_int(value: int, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} is outside the allowed range.")
    return value


def _bounded_float(value: float, minimum: float, maximum: float, label: str) -> float:
    converted = float(value)
    if not minimum <= converted <= maximum:
        raise ValueError(f"{label} is outside the allowed range.")
    return converted


def _safe_start_failure_code(exc: Exception) -> str:
    if isinstance(exc, SandboxUnavailableError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return "executable_not_found"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, ValueError):
        return "invalid_process_request"
    return "process_start_failed"


async def _quiet_task(task: asyncio.Task[Any]) -> None:
    with suppress(asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
        await task
