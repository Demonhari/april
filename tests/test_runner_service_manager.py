from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import IO

import pytest

from apps.runner.install import (
    APRIL_WRAPPER_MARKER,
    PATH_BLOCK_START,
    add_path_block,
    install_wrappers,
    path_contains_dir,
    wrapper_content,
)
from apps.runner.service_manager import AprilServiceManager


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class FakePopen:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.next_pid = 1000

    def __call__(
        self,
        args,
        *,
        cwd: str,
        env,
        stdout: IO[bytes],
        stderr: int,
        start_new_session: bool,
    ) -> FakeProcess:
        self.next_pid += 1
        self.calls.append(
            {
                "args": list(args),
                "cwd": cwd,
                "env": dict(env),
                "stderr": stderr,
                "start_new_session": start_new_session,
                "stdout": stdout,
            }
        )
        return FakeProcess(self.next_pid)


def test_status_reports_stopped_without_pid_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APRIL_HOME", raising=False)
    manager = AprilServiceManager(home=tmp_path)
    status = manager.status()
    assert status.runtime.running is False
    assert status.api.running is False


def test_stale_pid_files_are_ignored(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APRIL_HOME", raising=False)
    run_dir = tmp_path / "data" / "run"
    run_dir.mkdir(parents=True)
    stale = run_dir / "runtime.pid"
    stale.write_text("12345", encoding="utf-8")
    manager = AprilServiceManager(home=tmp_path, pid_exists=lambda _pid: False)
    status = manager.status()
    assert status.runtime.running is False
    assert not stale.exists()


def test_service_manager_uses_argv_arrays_and_fake_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APRIL_HOME", raising=False)
    popen = FakePopen()
    manager = AprilServiceManager(
        home=tmp_path,
        python_executable="/test/python",
        popen_factory=popen,
        health_getter=lambda _url, _timeout: True,
        pid_exists=lambda _pid: True,
    )
    status = manager.start(fake_backend=True)
    assert status.ok
    assert len(popen.calls) == 2
    runtime_call = popen.calls[0]
    assert runtime_call["args"] == ["/test/python", "-m", "services.april_runtime.server"]
    assert runtime_call["cwd"] == str(tmp_path)
    assert runtime_call["start_new_session"] is True
    assert runtime_call["env"]["APRIL_RUNTIME_BACKEND"] == "fake"
    assert runtime_call["env"]["APRIL_HOME"] == str(tmp_path)


def test_wrapper_content_includes_april_home(tmp_path: Path) -> None:
    content = wrapper_content(repo_root=tmp_path)
    assert APRIL_WRAPPER_MARKER in content
    assert f'export APRIL_HOME="{tmp_path.resolve()}"' in content
    assert ".venv/bin/python" in content
    assert "apps.runner.main" in content


def test_installer_creates_executable_wrappers(tmp_path: Path) -> None:
    bin_dir = tmp_path / ".local" / "bin"
    result = install_wrappers(repo_root=tmp_path, bin_dir=bin_dir)
    run_path = bin_dir / "run"
    april_run_path = bin_dir / "april-run"
    assert run_path in result.installed
    assert april_run_path in result.installed
    for path in (run_path, april_run_path):
        assert path.exists()
        assert os.access(path, os.X_OK)
        content = path.read_text(encoding="utf-8")
        assert APRIL_WRAPPER_MARKER in content
        assert f'export APRIL_HOME="{tmp_path.resolve()}"' in content
        assert ".venv/bin/python" in content
        assert "apps.runner.main" in content


def test_installer_refuses_non_april_run_without_force(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    run_path = bin_dir / "run"
    run_path.write_text("#!/bin/sh\necho other\n", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc_info:
        install_wrappers(repo_root=tmp_path, bin_dir=bin_dir)
    assert "not APRIL-owned" in str(exc_info.value)
    result = install_wrappers(repo_root=tmp_path, bin_dir=bin_dir, force=True)
    assert run_path in result.installed
    assert APRIL_WRAPPER_MARKER in run_path.read_text(encoding="utf-8")


def test_add_to_path_writes_zshrc_managed_block_once(tmp_path: Path) -> None:
    config_path, changed = add_path_block(shell="/bin/zsh", home=tmp_path)
    assert changed is True
    assert config_path == tmp_path / ".zshrc"
    content = config_path.read_text(encoding="utf-8")
    assert PATH_BLOCK_START in content
    assert 'export PATH="$HOME/.local/bin:$PATH"' in content
    _, changed_again = add_path_block(shell="/bin/zsh", home=tmp_path)
    assert changed_again is False
    assert config_path.read_text(encoding="utf-8").count(PATH_BLOCK_START) == 1


def test_path_contains_dir_detects_local_bin(tmp_path: Path) -> None:
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    assert path_contains_dir(local_bin, path_value=str(local_bin))
    assert not path_contains_dir(local_bin, path_value=str(tmp_path / "other"))


def test_logs_command_reads_runtime_and_api_logs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APRIL_HOME", raising=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "runtime.log").write_text("runtime one\nruntime two\n", encoding="utf-8")
    (logs / "api.log").write_text("api one\napi two\n", encoding="utf-8")
    manager = AprilServiceManager(home=tmp_path)
    output = manager.logs(lines=1)
    assert "runtime two" in output
    assert "api two" in output
    assert "runtime one" not in output


def test_stop_removes_pid_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APRIL_HOME", raising=False)
    run_dir = tmp_path / "data" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "runtime.pid").write_text("101", encoding="utf-8")
    (run_dir / "api.pid").write_text("202", encoding="utf-8")
    alive = {101, 202}
    signals: list[tuple[int, int]] = []

    def pid_exists(pid: int) -> bool:
        return pid in alive

    def pid_signal(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        if sig == signal.SIGTERM:
            alive.discard(pid)

    manager = AprilServiceManager(
        home=tmp_path,
        pid_exists=pid_exists,
        pid_signal=pid_signal,
        health_getter=lambda _url, _timeout: False,
        sleep=lambda _seconds: None,
    )
    status = manager.stop()
    assert status.runtime.running is False
    assert status.api.running is False
    assert not (run_dir / "runtime.pid").exists()
    assert not (run_dir / "api.pid").exists()
    assert (202, signal.SIGTERM) in signals
    assert (101, signal.SIGTERM) in signals


def test_start_uses_home_even_from_other_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APRIL_HOME", raising=False)
    outside = tmp_path / "outside"
    home = tmp_path / "april-home"
    outside.mkdir()
    home.mkdir()
    monkeypatch.chdir(outside)
    popen = FakePopen()
    manager = AprilServiceManager(
        home=home,
        python_executable="/test/python",
        popen_factory=popen,
        health_getter=lambda _url, _timeout: True,
        pid_exists=lambda _pid: True,
    )
    manager.start()
    assert all(call["cwd"] == str(home) for call in popen.calls)


def test_make_verify_global_uses_home_local_bin_directly() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert 'verify-global:\n\t"$(HOME)/.local/bin/run" april status' in makefile


def test_setup_mac_global_add_to_path_uses_temp_home(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    env["SHELL"] = "/bin/zsh"
    env["APRIL_SETUP_SKIP_PIP"] = "1"
    env["APRIL_INSTALL_SKIP_PIP"] = "1"
    result = subprocess.run(
        ["bash", "scripts/setup_mac.sh", "--base", "--global", "--add-to-path"],
        cwd=Path.cwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".local" / "bin" / "run").exists()
    assert PATH_BLOCK_START in (tmp_path / ".zshrc").read_text(encoding="utf-8")
