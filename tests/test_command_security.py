from __future__ import annotations

import pytest
import yaml

from april_common.errors import PermissionDeniedError, ValidationError
from skills.terminal import command_policy
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


def test_command_policy_can_use_trusted_worker_roots(settings_tmp, tmp_path) -> None:
    command, cwd, rule = validate_command(
        ["pytest"],
        tmp_path,
        allowed_roots=(tmp_path,),
    )
    assert command[-1].endswith("/pytest")
    assert cwd == tmp_path.resolve()
    assert rule.executable == "pytest"


def test_command_policy_rejects_empty_and_invalid_configured_commands(settings_tmp) -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_command([], settings_tmp.home)
    with pytest.raises(PermissionDeniedError, match="Subcommand"):
        validate_command(["ruff", "invalid"], settings_tmp.home)
    with pytest.raises(PermissionDeniedError, match="python -m module"):
        validate_command(["python", "-m", "pip"], settings_tmp.home)


def test_command_policy_rejects_file_cwd_and_missing_binary(settings_tmp, tmp_path, monkeypatch):
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(PermissionDeniedError, match="working directory"):
        validate_command(["pytest"], file_path, allowed_roots=(tmp_path,))
    monkeypatch.setattr(command_policy.shutil, "which", lambda _name: None)
    with pytest.raises(PermissionDeniedError, match="not found"):
        validate_command(["pytest"], tmp_path, allowed_roots=(tmp_path,))


def test_command_policy_builds_restricted_environment(settings_tmp) -> None:
    del settings_tmp
    assert "HTTP_PROXY" not in command_policy.clean_environment()


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
