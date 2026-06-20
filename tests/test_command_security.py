from __future__ import annotations

import pytest
import yaml

from april_common.errors import PermissionDeniedError
from skills.terminal.command_policy import run_restricted_command, validate_command


def test_shell_metacharacters_rejected(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_command(["python", "-m", "timeit", "x|y"], settings_tmp.home)


def test_unapproved_command_rejected(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_command(["sh", "-c", "echo hi"], settings_tmp.home)


def test_command_allowlist_is_loaded_from_tools_yaml(settings_tmp) -> None:
    config_dir = settings_tmp.home / "configs"
    config_dir.mkdir()
    (config_dir / "tools.yaml").write_text(
        yaml.safe_dump(
            {
                "tools": {
                    "command_allowlist": [
                        {
                            "executable": "pytest",
                            "subcommands": [],
                            "permission_level": 3,
                            "risk_level": "code_write",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionDeniedError):
        validate_command(["ruff", "check"], settings_tmp.home)


def test_python_pip_install_rejected(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_command(["python", "-m", "pip", "install", "x"], settings_tmp.home)


def test_python_ensurepip_rejected(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_command(["python", "-m", "ensurepip"], settings_tmp.home)


def test_bash_c_rejected(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_command(["bash", "-c", "echo hi"], settings_tmp.home)


def test_command_substitution_rejected(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_command(["python", "-m", "timeit", "$(pwd)"], settings_tmp.home)


def test_pipes_and_redirection_rejected(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        validate_command(["python", "-m", "timeit", "x > y"], settings_tmp.home)
    with pytest.raises(PermissionDeniedError):
        validate_command(["python", "-m", "timeit", "x|y"], settings_tmp.home)


def test_executable_path_outside_allowlist_rejected(settings_tmp, tmp_path) -> None:
    executable = tmp_path / "pytest"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    with pytest.raises(PermissionDeniedError):
        validate_command([str(executable)], settings_tmp.home)


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
