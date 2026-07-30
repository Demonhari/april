from __future__ import annotations

import asyncio
import base64
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from april_common.audit import AuditLogger
from april_common.process_environment import ProcessCategory, build_process_environment
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    run_restricted_process,
)
from april_common.project_scope import (
    git_staged_digest,
    git_staged_tree_id,
    inspect_patch_bytes,
)
from services.jobs.registry import default_job_registry
from services.jobs.schemas import JobStatus
from services.jobs.store import JobStore, JobTransitionError
from services.jobs.worker import JobWorker
from services.memory.database import Database
from services.memory.migrations import SCHEMA_VERSION, run_migrations
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest
from services.tool_worker.client import ToolWorkerClient
from services.tool_worker.executor import ToolWorkerExecutor
from services.tool_worker.limits import (
    prepare_runtime_directory,
    write_capability_file,
)
from services.tool_worker.schemas import ToolWorkerRequest, ToolWorkerResponse
from services.tool_worker.server import ToolWorkerServer


@pytest.mark.asyncio
async def test_schema_19_job_store_claim_cancel_and_terminal_protection(tmp_path: Path) -> None:
    database = Database(tmp_path / "jobs.db")
    await database.connect()
    await run_migrations(database)
    try:
        assert SCHEMA_VERSION == 20
        store = JobStore(database, default_job_registry())
        job = await store.submit(job_type="self_check", payload={}, owner="local-user")
        first, second = await asyncio.gather(
            store.claim_next(worker_id="worker-a", lease_seconds=30),
            store.claim_next(worker_id="worker-b", lease_seconds=30),
        )
        claims = [claim for claim in (first, second) if claim is not None]
        assert len(claims) == 1
        claimed = claims[0]
        assert claimed.id == job.id
        assert claimed.attempt_count == 1
        await store.heartbeat(
            job.id,
            worker_id=str(claimed.worker_id),
            lease_seconds=30,
            progress_percent=25,
            progress_code="quarter",
        )
        cancelling, already_terminal = await store.request_cancel(job.id)
        assert cancelling.status is JobStatus.CANCELLING
        assert already_terminal is False
        completed = await store.finish(
            job.id,
            worker_id=str(claimed.worker_id),
            status=JobStatus.CANCELLED,
            error_code="cancelled",
        )
        assert completed.status is JobStatus.CANCELLED
        with pytest.raises(JobTransitionError, match="terminal_transition_denied"):
            await store.finish(
                job.id,
                worker_id=str(claimed.worker_id),
                status=JobStatus.SUCCEEDED,
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_expired_lease_requeues_restart_safe_job_and_preserves_nonretryable(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "jobs.db")
    await database.connect()
    await run_migrations(database)
    try:
        store = JobStore(database, default_job_registry())
        retryable = await store.submit(job_type="self_check", payload={}, owner="local-user")
        claim = await store.claim_next(worker_id="worker-a", lease_seconds=30)
        assert claim is not None
        async with database.transaction() as connection:
            await connection.execute(
                "UPDATE background_jobs SET lease_expires_at = '2000-01-01T00:00:00Z' WHERE id = ?",
                (retryable.id,),
            )
        recovered = await store.recover_expired_leases()
        assert recovered[0].status is JobStatus.QUEUED
        retry_claim = await store.claim_next(worker_id="worker-retry", lease_seconds=30)
        assert retry_claim is not None
        await store.finish(
            retry_claim.id,
            worker_id="worker-retry",
            status=JobStatus.SUCCEEDED,
        )

        limited = await store.submit(job_type="self_check", payload={}, owner="local-user")
        claim = await store.claim_next(worker_id="worker-b", lease_seconds=30)
        assert claim is not None
        async with database.transaction() as connection:
            await connection.execute(
                """
                UPDATE background_jobs
                SET lease_expires_at = '2000-01-01T00:00:00Z',
                    attempt_count = maximum_attempts
                WHERE id = ?
                """,
                (limited.id,),
            )
        recovered = await store.recover_expired_leases()
        assert recovered[0].status is JobStatus.INTERRUPTED
        assert recovered[0].error_code == "lease_expired"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_job_store_rejects_secrets_and_unapproved_test_job(tmp_path: Path) -> None:
    database = Database(tmp_path / "jobs.db")
    await database.connect()
    await run_migrations(database)
    try:
        store = JobStore(database, default_job_registry())
        with pytest.raises(ValueError, match="unsafe_structured_field"):
            await store.submit(
                job_type="repository_index",
                payload={"repo_path": str(tmp_path), "token": "sentinel"},
                owner="local-user",
            )
        with pytest.raises(JobTransitionError, match="approval_required"):
            await store.submit(
                job_type="configured_test",
                payload={"argv": ["pytest"], "cwd": str(tmp_path)},
                owner="local-user",
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_job_event_retention_and_retry_eligibility(tmp_path: Path) -> None:
    database = Database(tmp_path / "jobs.db")
    await database.connect()
    await run_migrations(database)
    try:
        store = JobStore(database, default_job_registry())
        job = await store.submit(job_type="self_check", payload={}, owner="local-user")
        claim = await store.claim_next(worker_id="worker", lease_seconds=30)
        assert claim is not None
        for index in range(110):
            await store.heartbeat(
                job.id,
                worker_id="worker",
                lease_seconds=30,
                progress_percent=min(index, 99),
                progress_code=f"progress_{index}",
            )
        assert len(await store.events(job.id)) == 100
        failed = await store.finish(
            job.id,
            worker_id="worker",
            status=JobStatus.FAILED,
            error_code="fixture_failure",
        )
        retried, already_queued = await store.retry(failed.id)
        assert retried.status is JobStatus.QUEUED
        assert already_queued is False
        repeated, already_queued = await store.retry(failed.id)
        assert repeated.status is JobStatus.QUEUED
        assert already_queued is True
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_job_worker_claims_and_completes_self_check(
    settings_tmp,
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "worker.db")
    await database.connect()
    await run_migrations(database)
    try:
        store = JobStore(database, default_job_registry())
        job = await store.submit(job_type="self_check", payload={}, owner="local-user")
        worker = JobWorker(
            settings=settings_tmp,
            database=database,
            store=store,
            tool_worker=None,
            worker_id="worker-fixture",
            lease_seconds=5,
            status_path=tmp_path / "worker-status.json",
        )
        assert await worker.run_once() is True
        completed = await store.require(job.id)
        assert completed.status is JobStatus.SUCCEEDED
        assert completed.result == {"self_check": True}
        assert await worker.run_once() is False
    finally:
        await database.close()


def test_process_environments_are_explicit_and_service_tokens_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sentinels = {
        "APRIL_RUNTIME_TOKEN": "runtime-sentinel",
        "APRIL_API_TOKEN": "api-sentinel",
        "AWS_SECRET_ACCESS_KEY": "aws-sentinel",
        "OPENAI_API_KEY": "openai-sentinel",
        "GITHUB_TOKEN": "github-sentinel",
        "DATABASE_PASSWORD": "database-sentinel",
        "SSH_AUTH_SOCK": "ssh-sentinel",
        "HTTPS_PROXY": "proxy-sentinel",
    }
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)
    for category in (
        ProcessCategory.TOOL_WORKER,
        ProcessCategory.RESTRICTED_COMMAND,
        ProcessCategory.TEST_RUNNER,
        ProcessCategory.GIT,
        ProcessCategory.REPOSITORY_INDEXING,
        ProcessCategory.DOCUMENT_PROCESSING,
    ):
        environment = build_process_environment(category, april_home=tmp_path)
        assert not set(sentinels).intersection(environment)
    runtime = build_process_environment(ProcessCategory.RUNTIME, april_home=tmp_path)
    assert "APRIL_RUNTIME_TOKEN" not in runtime
    assert "APRIL_API_TOKEN" not in runtime
    core = build_process_environment(ProcessCategory.CORE_API, april_home=tmp_path)
    assert "APRIL_RUNTIME_TOKEN" not in core
    assert "APRIL_API_TOKEN" not in core
    assert "OPENAI_API_KEY" not in core


@pytest.mark.asyncio
async def test_process_runner_bounds_output_and_times_out_process_group(tmp_path: Path) -> None:
    output = await run_restricted_process(
        [sys.executable, "-c", "print('x' * 10000)"],
        cwd=tmp_path,
        category=ProcessCategory.TEST_RUNNER,
        timeout_seconds=5,
        max_stdout_bytes=100,
        max_stderr_bytes=100,
        resource_limit_profile=ResourceLimitProfile.NONE,
    )
    assert output.status is ProcessStatus.COMPLETED
    assert len(output.stdout.encode()) == 100
    assert output.stdout_truncated is True

    timeout = await run_restricted_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        category=ProcessCategory.TEST_RUNNER,
        timeout_seconds=0.05,
        resource_limit_profile=ResourceLimitProfile.NONE,
        termination_grace_seconds=0.05,
    )
    assert timeout.status is ProcessStatus.TIMED_OUT
    assert timeout.failure_code == "timeout"
    assert timeout.returncode is not None

    ignores_term = await run_restricted_process(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ],
        cwd=tmp_path,
        category=ProcessCategory.TEST_RUNNER,
        timeout_seconds=0.1,
        resource_limit_profile=ResourceLimitProfile.NONE,
        termination_grace_seconds=0.05,
    )
    assert ignores_term.status is ProcessStatus.TIMED_OUT
    assert ignores_term.returncode == -9


@pytest.mark.asyncio
async def test_process_runner_durable_cancellation_signal(tmp_path: Path) -> None:
    cancellation = asyncio.Event()
    execution = asyncio.create_task(
        run_restricted_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            category=ProcessCategory.TEST_RUNNER,
            timeout_seconds=10,
            cancellation_event=cancellation,
            resource_limit_profile=ResourceLimitProfile.NONE,
        )
    )
    await asyncio.sleep(0.05)
    cancellation.set()
    result = await execution
    assert result.status is ProcessStatus.CANCELLED
    assert result.failure_code == "cancelled"


@pytest.mark.asyncio
async def test_tool_worker_owner_only_socket_protocol_and_idempotency() -> None:
    home = Path(tempfile.mkdtemp(prefix="april-tw-", dir="/tmp"))
    project = home / "project"
    project.mkdir()
    runtime = prepare_runtime_directory(home / "runtime", april_home=home)
    capability_path = runtime / "capability"
    capability = secrets.token_urlsafe(32)
    write_capability_file(capability_path, capability, runtime_directory=runtime)
    socket_path = runtime / "worker.sock"
    server = ToolWorkerServer(
        april_home=home,
        socket_path=socket_path,
        capability_path=capability_path,
        allowed_roots=(home,),
    )
    await server.start()
    task = asyncio.create_task(server.serve_forever())
    try:
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        client = ToolWorkerClient(
            socket_path=socket_path,
            capability_path=capability_path,
            runtime_directory=runtime,
        )
        first = await client.execute(
            request_id="same-request",
            operation="self_check",
            project_root=project,
            args={},
            timeout_seconds=5,
        )
        second = await client.execute(
            request_id="same-request",
            operation="self_check",
            project_root=project,
            args={},
            timeout_seconds=5,
        )
        assert first == second
        assert first.ok is True
        assert stat.S_IMODE((runtime / "request-outcomes.json").stat().st_mode) == 0o600
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await server.close()

    restarted = ToolWorkerServer(
        april_home=home,
        socket_path=socket_path,
        capability_path=capability_path,
        allowed_roots=(home,),
    )
    await restarted.start()
    restarted_task = asyncio.create_task(restarted.serve_forever())
    try:
        duplicate = await client.execute(
            request_id="same-request",
            operation="self_check",
            project_root=project,
            args={},
            timeout_seconds=5,
        )
        assert duplicate.ok is False
        assert duplicate.failure_code == "duplicate_request_completed"
    finally:
        restarted_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await restarted_task
        await restarted.close()
        shutil.rmtree(home)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(repo),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        },
    )


@pytest.mark.asyncio
async def test_tool_worker_revalidates_and_executes_patch_and_git_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APRIL_HOME", str(tmp_path))
    monkeypatch.setenv("APRIL_ALLOWED_FILESYSTEM_ROOTS", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "april@example.invalid")
    _git(repo, "config", "user.name", "APRIL Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "file.txt").write_text("old\n", encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-m", "initial")

    capability = secrets.token_urlsafe(32)
    executor = ToolWorkerExecutor(allowed_roots=(tmp_path,), capability=capability)
    patch_bytes = (
        b"diff --git a/file.txt b/file.txt\n"
        b"--- a/file.txt\n"
        b"+++ b/file.txt\n"
        b"@@ -1 +1 @@\n"
        b"-old\n"
        b"+new\n"
    )
    artifact = await inspect_patch_bytes(patch_bytes=patch_bytes, repo_root=repo)
    patch_response = await executor.execute(
        ToolWorkerRequest(
            request_id="patch-1",
            capability=capability,
            operation="patch_applier",
            project_root=str(repo),
            args={
                "patch_base64": base64.b64encode(patch_bytes).decode("ascii"),
                "patch_sha256": artifact.patch_sha256,
                "patch_byte_length": artifact.patch_byte_length,
                "affected_paths": artifact.affected_paths,
                "repo_root": str(repo),
                "repo_state_digest": artifact.repo_state_digest,
            },
            timeout_seconds=10,
            max_stdout_bytes=10_000,
            max_stderr_bytes=10_000,
        )
    )
    assert patch_response.ok is True
    assert (repo / "file.txt").read_text(encoding="utf-8") == "new\n"

    _git(repo, "add", "file.txt")
    commit_response = await executor.execute(
        ToolWorkerRequest(
            request_id="commit-1",
            capability=capability,
            operation="git_commit",
            project_root=str(repo),
            args={
                "message": "approved change",
                "staged_diff_sha256": await git_staged_digest(repo),
                "staged_tree_id": await git_staged_tree_id(repo),
            },
            timeout_seconds=10,
            max_stdout_bytes=10_000,
            max_stderr_bytes=10_000,
        )
    )
    assert commit_response.ok is True
    assert len(str(commit_response.data["commit_hash"])) == 40

    rejected = await executor.execute(
        ToolWorkerRequest(
            request_id="bad-auth",
            capability="x" * 32,
            operation="self_check",
            project_root=str(repo),
            timeout_seconds=5,
            max_stdout_bytes=0,
            max_stderr_bytes=0,
        )
    )
    assert rejected.failure_code == "authentication_failed"


class _FakeJobToolWorker:
    async def execute(self, **kwargs: object) -> ToolWorkerResponse:
        return ToolWorkerResponse(
            request_id=str(kwargs["request_id"]),
            ok=True,
            returncode=0,
            status="completed",
        )


@pytest.mark.asyncio
async def test_job_worker_delegates_approved_configured_test(
    settings_tmp,
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "test-job.db")
    await database.connect()
    await run_migrations(database)
    try:
        store = JobStore(database, default_job_registry())
        approval_args = {"argv": ["pytest"], "repo_path": str(tmp_path)}
        approvals = ApprovalStore(
            database,
            AuditLogger(tmp_path / "audit.jsonl"),
            expiry_seconds=300,
        )
        approval = await approvals.create(
            ApprovalRequest(
                tool="test_runner",
                args=approval_args,
                agent="local-operator",
                permission_level=3,
                risk_level="code_write",
            ),
            actor="local-user",
            request_id="configured-test-create",
        )
        await approvals.approve_exact(
            approval_id=approval.approval_id,
            tool="test_runner",
            args=approval_args,
            actor="local-user",
            request_id="configured-test-approve",
        )
        job, created = await store.submit_with_exact_approval(
            job_type="configured_test",
            payload={"argv": ["pytest"], "cwd": str(tmp_path)},
            owner="local-user",
            approval_id=approval.approval_id,
            approval_tool="test_runner",
            approval_args=approval_args,
        )
        assert created is True
        worker = JobWorker(
            settings=settings_tmp,
            database=database,
            store=store,
            tool_worker=_FakeJobToolWorker(),  # type: ignore[arg-type]
            worker_id="test-worker",
            lease_seconds=5,
        )
        assert await worker.run_once() is True
        completed = await store.require(job.id)
        assert completed.status is JobStatus.SUCCEEDED
        assert completed.result == {
            "returncode": 0,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }
    finally:
        await database.close()
