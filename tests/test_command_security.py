from __future__ import annotations

import pytest

from april_common.errors import PermissionDeniedError
from skills.terminal.command_policy import run_restricted_command, validate_command


def test_shell_metacharacters_rejected(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_command(["python", "-m", "timeit", "x|y"], settings_tmp.home)


def test_unapproved_command_rejected(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_command(["sh", "-c", "echo hi"], settings_tmp.home)


@pytest.mark.asyncio
async def test_timeout_handled(settings_tmp) -> None:
    code, _stdout, stderr = await run_restricted_command(
        ["python", "-m", "timeit", "while True: pass"],
        settings_tmp.home,
        timeout=0.01,
    )
    assert code == 124
    assert "timed out" in stderr


@pytest.mark.asyncio
async def test_output_capped(settings_tmp) -> None:
    code, stdout, _stderr = await run_restricted_command(
        ["python", "-m", "timeit", "-n", "1", "'x' * 10"],
        settings_tmp.home,
        timeout=5,
    )
    assert code == 0
    assert len(stdout) <= 100_000
