from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from apps.runner.install import install_wrappers
from apps.runner.main import app
from apps.runner.service_manager import ServiceInfo, ServiceStatus


class FakeManager:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.started: list[bool] = []
        self.stopped = False
        self.restarted: list[bool] = []

    def _status(self, *, ok: bool = True) -> ServiceStatus:
        return ServiceStatus(
            runtime=ServiceInfo(
                name="runtime",
                pid=111 if ok else None,
                running=ok,
                healthy=ok,
                log_path=self.home / "logs" / "runtime.log",
            ),
            api=ServiceInfo(
                name="api",
                pid=222 if ok else None,
                running=ok,
                healthy=ok,
                log_path=self.home / "logs" / "api.log",
            ),
        )

    def start(self, *, fake_backend: bool = False) -> ServiceStatus:
        self.started.append(fake_backend)
        return self._status()

    def status(self) -> ServiceStatus:
        return self._status(ok=False)

    def stop(self) -> ServiceStatus:
        self.stopped = True
        return self._status(ok=False)

    def restart(self, *, fake_backend: bool = False) -> ServiceStatus:
        self.restarted.append(fake_backend)
        return self._status()

    def logs(self, *, lines: int = 80) -> str:
        return f"runtime log\napi log\nlines={lines}"


def test_run_april_status_does_not_start_services(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "status"])
    assert result.exit_code == 0
    assert manager.started == []
    assert "runtime" in result.output


def test_run_april_ask_delegates_after_services(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    delegated: list[list[str]] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main._run_april_cli", lambda args: delegated.append(args) or 0)
    result = CliRunner().invoke(app, ["april", "ask", "April, plan my work today."])
    assert result.exit_code == 0
    assert manager.started == [False]
    assert delegated == [["ask", "April, plan my work today."]]


def test_run_april_fake_reaches_service_start(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    delegated: list[list[str]] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr("apps.runner.main._run_april_cli", lambda args: delegated.append(args) or 0)
    result = CliRunner().invoke(app, ["april", "ask", "hello", "--fake"])
    assert result.exit_code == 0
    assert manager.started == [True]
    assert delegated == [["ask", "hello"]]


def test_run_april_logs_prints_recent_logs(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "logs", "--lines", "150"])
    assert result.exit_code == 0
    assert "runtime log" in result.output
    assert "lines=150" in result.output


def test_run_april_stop_calls_manager(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "stop"])
    assert result.exit_code == 0
    assert manager.stopped is True


def test_doctor_reports_missing_path(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    empty_bin = tmp_path / "empty-bin"
    home.mkdir()
    repo.mkdir()
    empty_bin.mkdir()
    install_wrappers(repo_root=repo, bin_dir=home / ".local" / "bin")
    manager = FakeManager(repo)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(empty_bin))
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "run was not found in PATH" in result.output
    assert 'export PATH="$HOME/.local/bin:$PATH"' in result.output
    assert "make install-global" in result.output


def test_doctor_reports_ok_when_path_contains_april_wrapper(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    local_bin = home / ".local" / "bin"
    install_wrappers(repo_root=repo, bin_dir=local_bin)
    manager = FakeManager(repo)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(local_bin))
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "doctor"])
    assert result.exit_code == 0
    assert "OK: run resolves to an APRIL wrapper visible in PATH" in result.output


def test_doctor_reports_non_april_run_command(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    local_bin = home / ".local" / "bin"
    home.mkdir()
    repo.mkdir()
    local_bin.mkdir(parents=True)
    other_run = local_bin / "run"
    other_run.write_text("#!/usr/bin/env bash\necho other\n", encoding="utf-8")
    other_run.chmod(0o755)
    manager = FakeManager(repo)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(local_bin))
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "run resolves to a non-APRIL command" in result.output
    assert "make install-global-force" in result.output
