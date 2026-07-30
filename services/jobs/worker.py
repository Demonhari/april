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

from apps.runner.commands.model_compare import _compare as run_model_setup_comparison
from april_common.audit import audit_logger_for_settings
from april_common.effective_config import load_agents_file
from april_common.settings import AprilSettings, load_settings
from april_common.time import utc_now_iso
from services.april_runtime.client import RuntimeClient
from services.evolution.rollouts import RealPromptShadowEvaluator, RolloutService
from services.jobs.finetune_job import FinetuneJobError, run_finetune_job
from services.jobs.model_import import (
    ModelImportError,
    reconcile_model_imports,
    run_model_import_job,
)
from services.jobs.model_jobs import ModelJobError, run_model_utility_job
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
        self._cancellation_events: dict[str, asyncio.Event] = {}

    async def run_forever(self, *, poll_seconds: float = 0.25) -> None:
        await asyncio.to_thread(reconcile_model_imports, self.settings)
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
        self._cancellation_events[job.id] = asyncio.Event()
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
        except Exception as exc:
            if not execution.done():
                execution.cancel()
                with suppress(asyncio.CancelledError):
                    await execution
            await self.store.finish(
                job.id,
                worker_id=self.worker_id,
                status=JobStatus.FAILED,
                error_code=_safe_job_error_code(exc),
            )
        finally:
            self._cancellation_events.pop(job.id, None)
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
            cancellation = self._cancellation_events[job.id]
            memory = SqliteMemory(self.database)
            vector = vector_memory_from_settings(
                self.settings,
                audit=audit_logger_for_settings(self.settings),
            )
            repository = MemoryRepository(
                memory, vector, audit=audit_logger_for_settings(self.settings)
            )
            before = vector.health()
            await self.store.heartbeat(
                job.id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
                progress_percent=5,
                progress_code="memory_reindex_staging",
            )
            loop = asyncio.get_running_loop()

            def reindex_progress(completed: int, total: int) -> None:
                if cancellation.is_set():
                    raise asyncio.CancelledError
                percent = 10 + int(80 * completed / max(1, total))
                loop.call_soon_threadsafe(
                    asyncio.create_task,
                    self.store.heartbeat(
                        job.id,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                        progress_percent=min(percent, 90),
                        progress_code="memory_reindex_embedding",
                    ),
                )

            count = await repository.rebuild(progress=reindex_progress)
            after = vector.health()
            return {
                "record_count": count,
                "vector_count": int(after.get("vector_count") or 0),
                "provider": vector.embedding.name,
                "dimensions": vector.embedding.dimensions,
                "model_id": getattr(vector.embedding, "model_id", None),
                "active_generation": before.get("active_generation"),
                "final_generation": after.get("active_generation"),
                "validation_result": {
                    "ok": bool(after.get("compatible")),
                    "failure_reasons": list(after.get("failure_reasons") or []),
                },
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
        if job.job_type == "model_import":
            cancellation = self._cancellation_events[job.id]

            async def import_progress(percent: int, code: str) -> None:
                await self.store.heartbeat(
                    job.id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                    progress_percent=percent,
                    progress_code=code,
                )

            return await run_model_import_job(
                self.settings,
                operation_id=job.id,
                payload=job.payload,
                cancellation_event=cancellation,
                progress=import_progress,
            )
        if job.job_type in {"model_import_verification", "model_benchmark"}:
            cancellation = self._cancellation_events[job.id]
            await self.store.heartbeat(
                job.id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
                progress_percent=5,
                progress_code="model_artifact_validation",
            )
            return await run_model_utility_job(
                self.settings,
                model_id=str(job.payload["model_id"]),
                mode="verify" if job.job_type == "model_import_verification" else "benchmark",
                cancellation_event=cancellation,
                timeout_seconds=(900.0 if job.job_type == "model_import_verification" else 3600.0),
            )
        if job.job_type == "model_setup_comparison":
            cancellation = self._cancellation_events[job.id]

            async def comparison_progress(
                percent: int,
                code: str,
                checkpoint: dict[str, Any],
            ) -> None:
                await self.store.checkpoint(
                    job.id,
                    worker_id=self.worker_id,
                    result=checkpoint,
                    progress_percent=percent,
                    progress_code=code,
                )

            return await run_model_setup_comparison(
                str(job.payload["shared_model_id"]),
                cooldown_seconds=float(job.payload.get("cooldown_seconds", 0.0)),
                settings=self.settings,
                cancellation_event=cancellation,
                progress=comparison_progress,
                resume=job.result,
            )
        if job.job_type == "finetune":
            cancellation = self._cancellation_events[job.id]

            async def report_progress(percent: int, code: str) -> None:
                await self.store.heartbeat(
                    job.id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                    progress_percent=percent,
                    progress_code=code,
                )

            return await run_finetune_job(
                self.settings,
                plan_id=str(job.payload["plan_id"]),
                cancellation_event=cancellation,
                progress=report_progress,
            )
        if job.job_type == "dream_cycle":
            if not self.settings.evolution.enabled:
                raise RuntimeError("dream_cycle_disabled")
            from april_common.time import utc_now
            from services.evolution.dreamer import run_standalone

            await self.store.heartbeat(
                job.id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
                progress_percent=5,
                progress_code="dream_cycle_gates",
            )
            dream_result = await run_standalone(self.settings, utc_now())
            return {
                "status": dream_result.status,
                "reason_code": _safe_reason_code(dream_result.reason),
                "activated_candidate": False,
            }
        if job.job_type == "evolution_shadow":
            cancellation = self._cancellation_events[job.id]
            record = await RolloutService(
                self.settings,
                self.database,
                audit=audit_logger_for_settings(self.settings),
            ).require(str(job.payload["rollout_id"]))
            agents = load_agents_file(self.settings.home).agents
            target = agents.get(record.target_id)
            judge = agents.get("reading_agent")
            await self.store.heartbeat(
                job.id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
                progress_percent=5,
                progress_code="shadow_baseline_candidate_ab",
            )
            rollout_result = await RolloutService(
                self.settings,
                self.database,
                audit=audit_logger_for_settings(self.settings),
            ).start_shadow(
                record.id,
                evaluator=RealPromptShadowEvaluator(
                    self.settings,
                    RuntimeClient(
                        self.settings.runtime.url,
                        timeout=self.settings.runtime.request_timeout_seconds,
                        token=self.settings.runtime.token,
                    ),
                    model_id=target.model_id if target is not None else None,
                    judge_model_id=judge.model_id if judge is not None else None,
                ),
                cancellation_event=cancellation,
            )
            return {
                "rollout_id": rollout_result.id,
                "state": rollout_result.state,
                "reason_code": rollout_result.reason_code,
                "completed_sample_count": rollout_result.completed_sample_count,
                "shadow_evidence_sha256": rollout_result.shadow_evidence_sha256,
            }
        raise RuntimeError("job_type_unavailable")

    async def _cancel_execution(
        self,
        job: ClaimedJob,
        execution: asyncio.Task[dict[str, Any]],
    ) -> None:
        cancellation = self._cancellation_events.get(job.id)
        if cancellation is not None:
            cancellation.set()
        if job.job_type == "configured_test" and self.tool_worker is not None:
            with suppress(ToolWorkerUnavailable):
                await self.tool_worker.cancel(
                    target_request_id=f"job:{job.id}",
                    project_root=Path(str(job.payload["cwd"])).expanduser().resolve(strict=True),
                )
        if job.job_type == "configured_test":
            execution.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(execution, timeout=10.0)

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


async def _run(args: argparse.Namespace) -> None:  # pragma: no cover
    settings = load_settings(root=Path(args.april_home))
    registry: JobRegistry = default_job_registry(
        finetune_enabled=settings.finetune.enabled,
        evolution_enabled=settings.evolution.enabled,
    )
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


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="APRIL durable Job Worker")
    parser.add_argument("--april-home", required=True)
    parser.add_argument("--status-file", required=True)
    asyncio.run(_run(parser.parse_args()))


def _safe_job_error_code(exc: Exception) -> str:
    if isinstance(exc, (FinetuneJobError, ModelImportError, ModelJobError)):
        value = str(exc)
        if value and len(value) <= 160 and value.replace("_", "").isalnum():
            return value
    value = str(exc)
    if value in {"dream_cycle_disabled", "job_type_unavailable"}:
        return value
    return "job_handler_failed"


def _safe_reason_code(value: str) -> str:
    normalized = "_".join(value.casefold().split())
    safe_value = "".join(
        character for character in normalized if character.isalnum() or character == "_"
    )
    return safe_value[:160]


if __name__ == "__main__":
    main()
