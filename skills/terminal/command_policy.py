from __future__ import annotations

import asyncio
import os
import shutil
import signal
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from april_common.errors import PermissionDeniedError, ValidationError
from april_common.path_security import normalize_existing_path
from april_common.settings import get_settings
from skills.filesystem.common import current_path_policy

SHELL_META = {"|", "&", ";", ">", "<", "$", "`", "$(", "&&", "||", "\n"}
MAX_COMMAND_OUTPUT = 100_000
ALLOWED_PYTHON_MODULES = {"timeit", "pytest", "ruff"}
DENIED_EXECUTABLES = {
    "bash",
    "brew",
    "chmod",
    "chown",
    "conda",
    "curl",
    "fish",
    "mv",
    "npm",
    "pip",
    "pip3",
    "pnpm",
    "rm",
    "sh",
    "yarn",
    "zsh",
}


@dataclass(frozen=True, slots=True)
class CommandRule:
    executable: str
    subcommands: tuple[str, ...] = field(default_factory=tuple)
    permission_level: int = 3
    risk_level: str = "code_write"


DEFAULT_RULES = {
    "pytest": CommandRule("pytest", permission_level=3),
    "ruff": CommandRule("ruff", subcommands=("check", "format"), permission_level=3),
    "python": CommandRule("python", subcommands=("-m",), permission_level=3),
}


def _reject_shell_meta(argv: list[str]) -> None:
    for arg in argv:
        if any(meta in arg for meta in SHELL_META):
            raise PermissionDeniedError("Shell metacharacters are denied.")


def validate_command(argv: list[str], cwd: str | Path) -> tuple[list[str], Path, CommandRule]:
    if not argv:
        raise ValidationError("Command argv cannot be empty.")
    _reject_shell_meta(argv)
    requested_executable = Path(argv[0])
    if requested_executable.name != argv[0]:
        raise PermissionDeniedError("Executable paths are not accepted.")
    executable = requested_executable.name
    if executable in DENIED_EXECUTABLES:
        raise PermissionDeniedError("Executable is explicitly denied.", {"executable": executable})
    rule = DEFAULT_RULES.get(executable)
    if rule is None:
        raise PermissionDeniedError("Executable is not allowlisted.", {"executable": executable})
    if rule.subcommands and (len(argv) < 2 or argv[1] not in rule.subcommands):
        raise PermissionDeniedError(
            "Subcommand is not allowlisted.",
            {"executable": executable, "allowed": list(rule.subcommands)},
        )
    if executable == "python" and (
        len(argv) < 3 or argv[1] != "-m" or argv[2] not in ALLOWED_PYTHON_MODULES
    ):
        raise PermissionDeniedError(
            "python -m module is not allowlisted.",
            {"allowed_modules": sorted(ALLOWED_PYTHON_MODULES)},
        )
    resolved = normalize_existing_path(cwd, current_path_policy())
    if not resolved.is_dir():
        raise PermissionDeniedError("Command working directory must be an allowed directory.")
    binary = shutil.which(argv[0])
    if binary is None:
        raise PermissionDeniedError("Executable was not found.", {"executable": argv[0]})
    return [binary, *argv[1:]], resolved, rule


def clean_environment() -> dict[str, str]:
    blocked = ("TOKEN", "SECRET", "PASSWORD", "AUTH", "KEY", "CREDENTIAL")
    return {
        key: value
        for key, value in os.environ.items()
        if not any(part in key.upper() for part in blocked)
    }


async def run_restricted_command(
    argv: list[str], cwd: str | Path, *, timeout: float | None = None
) -> tuple[int, str, str]:
    settings = get_settings()
    command, resolved_cwd, _rule = validate_command(argv, cwd)
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(resolved_cwd),
        env=clean_environment(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout or settings.permissions.tool_timeout_seconds,
        )
    except TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        await process.wait()
        return 124, "", "Command timed out."
    stdout = stdout_bytes.decode("utf-8", errors="replace")[:MAX_COMMAND_OUTPUT]
    stderr = stderr_bytes.decode("utf-8", errors="replace")[:MAX_COMMAND_OUTPUT]
    return process.returncode or 0, stdout, stderr
