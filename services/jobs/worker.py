from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings, load_settings
from april_common.time import utc_now_iso
from services.jobs.registry import JobRegistry, default_job_registry
from services.jobs.schemas import DEFAULT_LEASE_SECONDS, ClaimedJob, JobStatus
from services.jobs.store import JobStore, JobTransitionError
from services.memory.database import Database
from services.memory.factory import vector_memory_from_settings
from services.memory.migrations import run_migrations
from services.memory.repository import MemoryRepository
from services.memory.sqlite_memory import SqliteMemory
from services.tool_worker.client import ToolWorkerClient, ToolWorkerUnavailable
from services.tool_worker.limits import default_tool_worker_runtime_directory
from skills.code.repo_indexer import repo_indexer
from skills.documents.document_indexer import document_indexer

JOB_WORKER_STATUS_VERSION = 1


class JobWorker:
    def __init__(
        self,
        *,
        settings: AprilSettings,
        database: Database,
        store: JobStore,
        tool_worker: ToolWorkerClient | None,
        worker_id: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        status_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.store = store
        self.tool_worker = tool_worker
        self.worker_id = worker_id or f"job-worker-{uuid.uuid4()}"
        self.lease_seconds = lease_seconds
        self.status_path = status_path
        self._stopping = asyncio.Event()

    async def run_forever(self, *, poll_seconds: float = 0.25) -> None:
        await self.store.recover_expired_leases()
        self._write_status(ready=True, active_job_id=None)
        while not self._stopping.is_set():
            handled = await self.run_once()
            if not handled:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stopping.wait(), timeout=poll_seconds)
        self._write_status(ready=False, active_job_id=None)

    def stop(self) -> None:
        self._stopping.set()

    async def run_once(self) -> bool:
        job = await self.store.claim_next(
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            self._write_status(ready=True, active_job_id=None)
            return False
        self._write_status(ready=True, active_job_id=job.id)
        execution = asyncio.create_task(self._execute(job))
        try:
            while not execution.done():
                await asyncio.sleep(min(1.0, self.lease_seconds / 3))
                if await self.store.cancellation_requested(job.id, self.worker_id):
                    await self._cancel_execution(job, execution)
                    await self.store.finish(
                        job.id,
                        worker_id=self.worker_id,
                        status=JobStatus.CANCELLED,
                        error_code="cancelled",
                    )
                    return True
                await self.store.heartbeat(
                    job.id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
            result = await execution
            await self.store.finish(
                job.id,
                worker_id=self.worker_id,
                status=JobStatus.SUCCEEDED,
                result=result,
            )
        except asyncio.CancelledError:
            await self._cancel_execution(job, execution)
            raise
        except JobTransitionError:
            raise
        except Exception:
            if not execution.done():
                execution.cancel()
                with suppress(asyncio.CancelledError):
                    await execution
            await self.store.finish(
                job.id,
                worker_id=self.worker_id,
                status=JobStatus.FAILED,
                error_code="job_handler_failed",
            )
        finally:
            self._write_status(ready=True, active_job_id=None)
        return True

    async def _execute(self, job: ClaimedJob) -> dict[str, Any]:
        if job.job_type == "self_check":
            return {"self_check": True}
        if job.job_type == "repository_index":
            result = await repo_indexer(
                {
                    "repo_path": job.payload["repo_path"],
                    "project_id": job.payload.get("project_id"),
                    "force_full_reindex": bool(job.payload.get("force_full_reindex", False)),
                }
            )
            if not result.ok:
                raise RuntimeError("repository_index_failed")
            return {
                "reindexed_files": int(result.data.get("reindexed_files", 0)),
                "skipped_files": int(result.data.get("skipped_files", 0)),
                "removed_files": int(result.data.get("removed_files", 0)),
                "chunks": int(result.data.get("chunks", 0)),
            }
        if job.job_type == "document_index":
            result = await document_indexer({"folder_path": job.payload["folder_path"]})
            if not result.ok:
                raise RuntimeError("document_index_failed")
            return {
                "chunks": int(result.data.get("chunks", 0)),
                "document_count": len(result.data.get("documents", [])),
                "unsupported_count": len(result.data.get("unsupported", [])),
            }
        if job.job_type == "memory_reindex":
            memory = SqliteMemory(self.database)
            vector = vector_memory_from_settings(
                self.settings,
                audit=AuditLogger(self.settings.audit_path),
            )
            repository = MemoryRepository(
                memory, vector, audit=AuditLogger(self.settings.audit_path)
            )
            count = await repository.rebuild()
            return {
                "reindexed": count,
                "provider": vector.embedding.name,
                "dimensions": vector.embedding.dimensions,
            }
        if job.job_type == "configured_test":
            if self.tool_worker is None:
                raise ToolWorkerUnavailable("tool_worker_unavailable")
            root = Path(str(job.payload["cwd"])).expanduser().resolve(strict=True)
            response = await self.tool_worker.execute(
                request_id=f"job:{job.id}",
                operation="test_runner",
                project_root=root,
                args={"argv": list(job.payload["argv"])},
                timeout_seconds=1800.0,
            )
            if not response.ok:
                raise RuntimeError("configured_test_failed")
            return {
                "returncode": response.returncode,
                "stdout_truncated": response.stdout_truncated,
                "stderr_truncated": response.stderr_truncated,
            }
        raise RuntimeError("job_type_unavailable")

    async def _cancel_execution(
        self,
        job: ClaimedJob,
        execution: asyncio.Task[dict[str, Any]],
    ) -> None:
        if job.job_type == "configured_test" and self.tool_worker is not None:
            with suppress(ToolWorkerUnavailable):
                await self.tool_worker.cancel(
                    target_request_id=f"job:{job.id}",
                    project_root=Path(str(job.payload["cwd"])).expanduser().resolve(strict=True),
                )
        execution.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await execution

    def _write_status(self, *, ready: bool, active_job_id: str | None) -> None:
        if self.status_path is None:
            return
        self.status_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        payload = {
            "version": JOB_WORKER_STATUS_VERSION,
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "ready": ready,
            "active_job": active_job_id is not None,
            "updated_at": utc_now_iso(),
        }
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.status_path)


async def _run(args: argparse.Namespace) -> None:
    settings = load_settings(root=Path(args.april_home))
    registry: JobRegistry = default_job_registry()
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    runtime_directory = default_tool_worker_runtime_directory(settings.home)
    tool_client = ToolWorkerClient(
        socket_path=runtime_directory / "worker.sock",
        capability_path=runtime_directory / "capability",
        runtime_directory=runtime_directory,
    )
    worker = JobWorker(
        settings=settings,
        database=database,
        store=JobStore(database, registry),
        tool_worker=tool_client,
        status_path=Path(args.status_file),
    )
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, worker.stop)
    try:
        await worker.run_forever()
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="APRIL durable Job Worker")
    parser.add_argument("--april-home", required=True)
    parser.add_argument("--status-file", required=True)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
