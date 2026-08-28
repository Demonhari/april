from __future__ import annotations

import asyncio
import base64
import platform
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from april_common.process_environment import ProcessCategory
from april_common.process_runner import ProcessStatus, run_restricted_process
from april_common.process_sandbox import (
    HostProcessSandbox,
    NetworkPolicy,
    SandboxBackend,
    SandboxOperation,
    SandboxPolicy,
    SandboxUnavailableError,
    generate_seatbelt_profile,
    operation_policy,
)
from april_common.project_scope import inspect_patch_bytes
from services.tool_worker.executor import ToolWorkerExecutor
from services.tool_worker.schemas import ToolWorkerRequest


def _policy(root: Path, *, network: NetworkPolicy = NetworkPolicy.DENY_ALL) -> SandboxPolicy:
    return SandboxPolicy(
        operation=SandboxOperation.RESTRICTED_COMMAND,
        network=network,
        readable_roots=(root,),
        writable_roots=(root,),
    )


def test_seatbelt_profile_denies_network_and_fences_roots(tmp_path: Path) -> None:
    profile = generate_seatbelt_profile(_policy(tmp_path), executable=sys.executable)
    assert "(deny network*)" in profile
    assert f'(allow file-read* (subpath "{tmp_path}"))' in profile
    assert f'(allow file-write* (subpath "{tmp_path}"))' in profile
    assert '(allow file-read-data (literal "/"))' in profile
    assert '(allow file-write* (literal "/dev/null"))' in profile
    assert '(allow file-read* (subpath "/usr/local/Cellar"))' in profile
    assert '(allow file-read* (subpath "/opt/homebrew/Cellar"))' in profile
    assert '(allow file-read* (subpath "/usr/local/etc"))' not in profile
    assert f'(deny file-read* (subpath "{tmp_path / ".ssh"}"))' in profile
    assert f'(deny file-read* (subpath "{tmp_path / "Library/Keychains"}"))' in profile


def test_operation_policy_selects_truthful_network_modes(tmp_path: Path) -> None:
    denied = operation_policy(
        ProcessCategory.TEST_RUNNER,
        project_root=tmp_path,
    )
    loopback = operation_policy(
        ProcessCategory.MODEL_VERIFICATION,
        project_root=tmp_path,
    )
    assert denied is not None
    assert denied.network is NetworkPolicy.DENY_ALL
    assert loopback is not None
    assert loopback.network is NetworkPolicy.LOOPBACK_ONLY


def test_sensitive_allowed_root_is_rejected(tmp_path: Path) -> None:
    sensitive = tmp_path / ".ssh"
    sensitive.mkdir()
    with pytest.raises(Exception, match="sensitive"):
        operation_policy(ProcessCategory.RESTRICTED_COMMAND, project_root=sensitive)


def test_production_fails_closed_without_supported_backend(tmp_path: Path) -> None:
    provider = HostProcessSandbox(system="Linux", sandbox_exec=tmp_path / "missing")
    report = provider.capabilities(environment="production", development_override=True)
    assert report.backend is SandboxBackend.UNAVAILABLE
    assert report.production_fail_closed is True
    with pytest.raises(SandboxUnavailableError, match="production_fail_closed"):
        provider.wrap(
            ["true"],
            policy=_policy(tmp_path),
            environment="production",
            development_override=True,
        )


def test_development_override_is_explicit_and_warns(tmp_path: Path) -> None:
    provider = HostProcessSandbox(system="Linux", sandbox_exec=tmp_path / "missing")
    unavailable = provider.capabilities(
        environment="development",
        development_override=False,
    )
    overridden = provider.capabilities(
        environment="development",
        development_override=True,
    )
    assert unavailable.backend is SandboxBackend.UNAVAILABLE
    assert overridden.backend is SandboxBackend.DEVELOPMENT_OVERRIDE
    assert overridden.network_denial_available is False
    assert overridden.warning is not None
    assert "DEVELOPMENT ONLY" in overridden.warning


@pytest.mark.asyncio
async def test_linux_development_override_runs_fake_restricted_process(tmp_path: Path) -> None:
    provider = HostProcessSandbox(system="Linux", sandbox_exec=tmp_path / "missing")
    result = await run_restricted_process(
        [sys.executable, "-c", "print('development-only')"],
        cwd=tmp_path,
        category=ProcessCategory.RESTRICTED_COMMAND,
        timeout_seconds=5,
        sandbox_policy=_policy(tmp_path),
        sandbox_environment="development",
        development_unsandboxed_override=True,
        sandbox_provider=provider,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "development-only"
    assert result.sandbox is not None
    assert result.sandbox.backend is SandboxBackend.DEVELOPMENT_OVERRIDE
    assert result.sandbox.network_denial_available is False
    assert result.sandbox.filesystem_policy_available is False


class _FakeProcess:
    def __init__(self) -> None:
        self.pid = 987_654
        self.returncode: int | None = 0
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(b"ok")
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.stdin = None

    async def wait(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_launcher_wraps_argv_without_shell_and_scrubs_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox_exec = tmp_path / "sandbox-exec"
    sandbox_exec.write_text("", encoding="utf-8")
    sandbox_exec.chmod(0o700)
    provider = HostProcessSandbox(system="Darwin", sandbox_exec=sandbox_exec)
    captured: dict[str, Any] = {}

    async def launcher(*argv: str, **kwargs: Any) -> Any:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeProcess()

    for key in (
        "APRIL_API_TOKEN",
        "APRIL_RUNTIME_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(key, "must-not-leak")
    result = await run_restricted_process(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        category=ProcessCategory.RESTRICTED_COMMAND,
        timeout_seconds=5,
        sandbox_policy=_policy(tmp_path),
        sandbox_provider=provider,
        process_launcher=launcher,
    )
    assert result.status is ProcessStatus.COMPLETED
    assert captured["argv"][0] == str(sandbox_exec)
    assert captured["argv"][-3:] == (sys.executable, "-c", "print('ok')")
    kwargs = captured["kwargs"]
    assert "shell" not in kwargs
    assert kwargs["start_new_session"] is True
    assert not {
        "APRIL_API_TOKEN",
        "APRIL_RUNTIME_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
    }.intersection(kwargs["env"])


@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS Seatbelt integration")
async def test_macos_seatbelt_denies_outbound_socket_when_operational(tmp_path: Path) -> None:
    provider = HostProcessSandbox()
    policy = _policy(tmp_path)
    probe = await run_restricted_process(
        [sys.executable, "-c", "print('seatbelt-probe')"],
        cwd=tmp_path,
        category=ProcessCategory.RESTRICTED_COMMAND,
        timeout_seconds=5,
        sandbox_policy=policy,
        sandbox_provider=provider,
    )
    if probe.returncode == 71 and "sandbox_apply" in probe.stderr:
        pytest.skip("sandbox-exec exists but the enclosing test sandbox denies profile application")
    assert probe.returncode == 0, probe.stderr
    denied = await run_restricted_process(
        [
            sys.executable,
            "-c",
            "import socket; socket.socket().connect(('1.1.1.1', 80))",
        ],
        cwd=tmp_path,
        category=ProcessCategory.RESTRICTED_COMMAND,
        timeout_seconds=5,
        sandbox_policy=policy,
        sandbox_provider=provider,
    )
    assert denied.returncode != 0


@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS Seatbelt integration")
async def test_macos_seatbelt_exact_patch_inspect_check_apply(tmp_path: Path) -> None:
    provider = HostProcessSandbox()
    probe = await run_restricted_process(
        [sys.executable, "-c", "print('seatbelt-probe')"],
        cwd=tmp_path,
        category=ProcessCategory.RESTRICTED_COMMAND,
        timeout_seconds=5,
        sandbox_policy=_policy(tmp_path),
        sandbox_provider=provider,
    )
    if probe.returncode == 71 and "sandbox_apply" in probe.stderr:
        pytest.skip("sandbox-exec exists but the enclosing test sandbox denies profile application")
    assert probe.returncode == 0, probe.stderr

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file.txt").write_text("old\n", encoding="utf-8")
    for args in (
        ("init",),
        ("config", "user.email", "april@example.invalid"),
        ("config", "user.name", "APRIL Test"),
        ("add", "file.txt"),
        ("commit", "-m", "initial"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    patch = (
        b"diff --git a/file.txt b/file.txt\n"
        b"--- a/file.txt\n"
        b"+++ b/file.txt\n"
        b"@@ -1 +1 @@\n"
        b"-old\n"
        b"+new\n"
    )
    artifact = await inspect_patch_bytes(
        patch_bytes=patch,
        repo_root=repo,
        sandbox_environment="development",
    )
    capability = secrets.token_urlsafe(32)
    response = await ToolWorkerExecutor(
        allowed_roots=(repo,),
        capability=capability,
        environment="development",
    ).execute(
        ToolWorkerRequest(
            request_id="mac-seatbelt-patch",
            capability=capability,
            operation="patch_applier",
            project_root=str(repo),
            args={
                "patch_base64": base64.b64encode(patch).decode("ascii"),
                "patch_sha256": artifact.patch_sha256,
                "patch_byte_length": artifact.patch_byte_length,
                "affected_paths": artifact.affected_paths,
                "repo_root": str(repo),
                "repo_state_digest": artifact.repo_state_digest,
            },
            timeout_seconds=15,
            max_stdout_bytes=4096,
            max_stderr_bytes=4096,
        )
    )
    assert response.ok is True, response
    assert (repo / "file.txt").read_text(encoding="utf-8") == "new\n"
