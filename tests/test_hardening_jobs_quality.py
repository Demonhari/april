from __future__ import annotations

import asyncio
import json
import secrets
import shutil
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from apps.runner.commands import model_compare
from apps.runner.main import app
from april_common.audit import AuditLogger
from services.april_runtime.schemas import ChatResponse, Usage
from services.evaluation import model_quality
from services.jobs.registry import default_job_registry
from services.jobs.schemas import JobStatus
from services.jobs.store import JobStore, JobTransitionError
from services.jobs.worker import JobWorker
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest
from services.tool_worker.executor import ToolWorkerExecutor
from services.tool_worker.schemas import ToolWorkerRequest, ToolWorkerResponse


def _import_payload(tmp_path: Path) -> dict[str, Any]:
    return {
        "source_path": str(tmp_path / "candidate.gguf"),
        "model_id": "candidate-brain",
        "role": "brain",
        "name": "Candidate",
        "expected_sha256": "a" * 64,
        "source_identity": {
            "device": 1,
            "inode": 2,
            "size": 4,
            "modified_ns": 3,
        },
        "format": "gguf",
        "destination": "models/candidate.gguf",
        "requested_verification": True,
    }


async def _approval(
    database: Database,
    tmp_path: Path,
    payload: dict[str, Any],
) -> tuple[ApprovalStore, str]:
    approvals = ApprovalStore(
        database,
        AuditLogger(tmp_path / "audit.jsonl"),
        expiry_seconds=300,
    )
    response = await approvals.create(
        ApprovalRequest(
            tool="model_import",
            args=payload,
            agent="local-operator",
            permission_level=4,
            risk_level="system_action",
            affected_paths=["candidate.gguf", "configs/models.yaml"],
            expected_side_effects=["register inactive model"],
        ),
        actor="local-user",
        request_id="approval-request",
    )
    return approvals, response.approval_id


@pytest.mark.asyncio
async def test_import_approval_consumption_and_job_acceptance_are_atomic(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "jobs.db")
    await database.connect()
    await run_migrations(database)
    try:
        payload = _import_payload(tmp_path)
        approvals, approval_id = await _approval(database, tmp_path, payload)
        store = JobStore(database, default_job_registry())
        with pytest.raises(JobTransitionError, match="approval_not_approved"):
            await store.submit_with_exact_approval(
                job_type="model_import",
                payload=payload,
                owner="local-user",
                approval_id=approval_id,
                approval_tool="model_import",
                approval_args=payload,
            )
        await approvals.approve_exact(
            approval_id=approval_id,
            tool="model_import",
            args=payload,
            actor="local-user",
            request_id="explicit-exact-approval",
        )
        outcomes = await asyncio.gather(
            *[
                store.submit_with_exact_approval(
                    job_type="model_import",
                    payload=payload,
                    owner="local-user",
                    approval_id=approval_id,
                    approval_tool="model_import",
                    approval_args=payload,
                )
                for _ in range(2)
            ]
        )
        assert outcomes[0][0].id == outcomes[1][0].id
        assert sorted(created for _job, created in outcomes) == [False, True]
        assert len(await store.list()) == 1
        assert (await approvals.get(approval_id)).status == "consumed"

        changed = {**payload, "requested_verification": False}
        with pytest.raises(JobTransitionError, match="approval_action_mismatch"):
            await store.submit_with_exact_approval(
                job_type="model_import",
                payload=changed,
                owner="local-user",
                approval_id=approval_id,
                approval_tool="model_import",
                approval_args=changed,
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_failed_job_acceptance_rolls_back_approval_consumption(tmp_path: Path) -> None:
    class FailingStore(JobStore):
        async def _append_event_tx(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("injected_event_failure")

    database = Database(tmp_path / "jobs.db")
    await database.connect()
    await run_migrations(database)
    try:
        payload = _import_payload(tmp_path)
        approvals, approval_id = await _approval(database, tmp_path, payload)
        await approvals.approve_exact(
            approval_id=approval_id,
            tool="model_import",
            args=payload,
            actor="local-user",
            request_id="explicit-exact-approval",
        )
        store = FailingStore(database, default_job_registry())
        with pytest.raises(RuntimeError, match="injected_event_failure"):
            await store.submit_with_exact_approval(
                job_type="model_import",
                payload=payload,
                owner="local-user",
                approval_id=approval_id,
                approval_tool="model_import",
                approval_args=payload,
            )
        assert (await approvals.get(approval_id)).status == "approved"
        assert await store.list() == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_comparison_checkpoint_survives_restart_recovery(
    settings_tmp: Any,
) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    try:
        store = JobStore(database, default_job_registry())
        job = await store.submit(
            job_type="model_setup_comparison",
            payload={"shared_model_id": "shared"},
            owner="local-user",
        )
        claimed = await store.claim_next(worker_id="comparison-worker", lease_seconds=30)
        assert claimed is not None
        checkpoint = {
            "checkpoint_type": "model_setup_comparison",
            "completed_results": {"specialist-a": {"passed": True}},
            "completed_model_ids": ["specialist-a"],
        }
        await store.checkpoint(
            job.id,
            worker_id="comparison-worker",
            result=checkpoint,
            progress_percent=45,
            progress_code="specialist_completed",
        )
        async with database.transaction() as connection:
            await connection.execute(
                "UPDATE background_jobs SET lease_expires_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", job.id),
            )
        recovered = await store.recover_expired_leases()
        assert recovered[0].status is JobStatus.QUEUED
        assert recovered[0].result == checkpoint
        assert store.registry.require("model_setup_comparison").cancellation_behavior == (
            "process_group"
        )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_comparison_cancellation_reaches_running_measurement(
    settings_tmp: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_compare(
        _shared_model_id: str,
        *,
        cancellation_event: asyncio.Event,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        started.set()
        await cancellation_event.wait()
        cancelled.set()
        raise asyncio.CancelledError

    monkeypatch.setattr("services.jobs.worker.run_model_setup_comparison", fake_compare)
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    try:
        store = JobStore(database, default_job_registry())
        job = await store.submit(
            job_type="model_setup_comparison",
            payload={"shared_model_id": "shared"},
            owner="local-user",
        )
        worker = JobWorker(
            settings=settings_tmp,
            database=database,
            store=store,
            tool_worker=None,
            lease_seconds=3,
        )
        execution = asyncio.create_task(worker.run_once())
        await asyncio.wait_for(started.wait(), timeout=2)
        await store.request_cancel(job.id)
        assert await asyncio.wait_for(execution, timeout=5) is True
        assert cancelled.is_set()
        assert (await store.require(job.id)).status is JobStatus.CANCELLED
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_comparison_recovery_reuses_completed_model_measurement(
    settings_tmp: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = settings_tmp.home / "configs" / "models.yaml"
    models.parent.mkdir(parents=True, exist_ok=True)
    models.write_text(
        """
models:
  specialist:
    id: specialist
    name: Specialist
    path: models/specialist.gguf
    backend: llama_cpp
    role: coding
    threads: 2
    context_size: 1024
    temperature: 0.0
    max_output_tokens: 128
  shared:
    id: shared
    name: Shared
    path: models/shared.gguf
    backend: llama_cpp
    role: brain
    threads: 2
    context_size: 1024
    temperature: 0.0
    max_output_tokens: 128
""",
        encoding="utf-8",
    )

    def measurement(model_id: str, role: str) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "role": role,
            "model_basename": f"{model_id}.gguf",
            "model_sha256": "a" * 64,
            "model_size": 4,
            "passed": True,
            "simulated": True,
            "fixture_set": {"installed": True, "sha256": "fixture"},
            "routing_accuracy": 1.0,
            "strict_json_first_pass_reliability": 1.0,
            "structured_json_reliability": 1.0,
            "coding_fixture_pass_rate": 1.0,
            "context_handling_reliability": 1.0,
            "lifecycle": {"model_switch_time_seconds": 1.0},
            "runs": [
                {
                    "run_index": 1,
                    "ok": True,
                    "load_time_seconds": 1.0,
                    "first_token_latency_seconds": 1.0,
                    "tokens_per_second": 10.0,
                    "prompt_token_count": 10,
                    "prompt_eval_duration_seconds": 1.0,
                    "process_rss_bytes": 100,
                    "peak_process_rss_bytes": 100,
                    "unload_success": True,
                },
                {
                    "run_index": 2,
                    "ok": True,
                    "load_time_seconds": 1.0,
                    "first_token_latency_seconds": 1.0,
                    "tokens_per_second": 10.0,
                    "prompt_token_count": 10,
                    "prompt_eval_duration_seconds": 1.0,
                    "process_rss_bytes": 100,
                    "peak_process_rss_bytes": 100,
                    "unload_success": True,
                },
            ],
        }

    calls: list[str] = []

    async def fake_measure(
        _settings: Any,
        *,
        model_id: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(model_id)
        return measurement(model_id, "brain")

    monkeypatch.setattr(model_compare, "run_model_utility_job", fake_measure)
    report = await model_compare._compare(
        "shared",
        settings=settings_tmp,
        resume={
            "completed_results": {
                "specialist": model_compare._redacted_benchmark(measurement("specialist", "coding"))
            }
        },
    )
    assert calls == ["shared"]
    assert report["current_specialist_configuration"]["models"][0]["model_id"] == ("specialist")


class _FixtureRuntime:
    fixtures: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def chat(self, *, request_id: str, model_id: str, **_kwargs: Any) -> ChatResponse:
        content: str
        if request_id.startswith("benchmark-route-"):
            fixture_id = request_id.removeprefix("benchmark-route-")
            fixture = self.fixtures[fixture_id]
            risk = fixture["risks"][0]
            content = json.dumps(
                {
                    "intent": fixture["category"],
                    "agent": fixture["agents"][0],
                    "model_id": model_id,
                    "confidence": 0.9,
                    "high_stakes": risk in {"system_action", "external_action"},
                    "tools_needed": fixture.get("tools_any", [])[:1],
                    "planned_tool_calls": [],
                    "memory_queries": [],
                    "permission_level": fixture.get("permission_min", 1),
                    "risk_level": risk,
                    "needs_confirmation": fixture.get("permission_min", 1) >= 3,
                    "task_steps": [],
                    "decision_summary": "fixture route",
                }
            )
        elif request_id == "benchmark-json-route-decision":
            content = '{"intent":"general","agent":"general_agent","risk_level":"none"}'
        elif request_id == "benchmark-json-tool-request":
            content = '{"type":"tool_request","tool":"read_file","args":{}}'
        elif request_id == "benchmark-json-final-answer":
            content = '{"type":"final_answer","message":"ok"}'
        elif request_id == "benchmark-code-clamp":
            content = json.dumps(
                {
                    "filename": "solution.py",
                    "content": "def clamp(value, low, high):\n"
                    "    return max(low, min(value, high))\n",
                }
            )
        elif request_id == "benchmark-code-unique_ordered":
            content = json.dumps(
                {
                    "filename": "solution.py",
                    "content": "def unique_ordered(values):\n"
                    "    return list(dict.fromkeys(values))\n",
                }
            )
        else:
            expected = {
                "benchmark-context-early-recall": "ORCHID-731",
                "benchmark-context-recent-instruction": "COBALT-19",
                "benchmark-context-tool-group": "4821",
                "benchmark-context-distraction": "MAPLE-552",
                "benchmark-context-structured-long": '{"owner":"Dev","status":"closed"}',
                "benchmark-context-near-limit": "QUARTZ-884",
            }
            content = expected[request_id]
        tokens = 900 if request_id.endswith("near-limit") else 64
        return ChatResponse(
            request_id=request_id,
            model_id=model_id,
            content=content,
            usage=Usage(input_tokens=tokens, output_tokens=8, total_tokens=tokens + 8),
        )


class _FixtureToolWorker:
    async def execute(self, **kwargs: Any) -> ToolWorkerResponse:
        return ToolWorkerResponse(
            request_id=str(kwargs["request_id"]),
            ok=True,
            returncode=0,
            status="completed",
            data={
                "forbidden_file_modification": False,
                "syntax_or_compilation_failure": False,
                "unnecessary_change": False,
            },
        )


@pytest.mark.asyncio
async def test_quality_worker_populates_real_model_metrics_without_router_shortcut(
    settings_tmp: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path.cwd() / "data" / "evaluations" / "model_benchmark" / "v1"
    target = settings_tmp.home / "data" / "evaluations" / "model_benchmark" / "v1"
    shutil.copytree(source, target)
    routing = json.loads((target / "routing.json").read_text(encoding="utf-8"))
    _FixtureRuntime.fixtures = {item["id"]: item for item in routing["fixtures"]}
    monkeypatch.setattr(model_quality, "RuntimeClient", _FixtureRuntime)

    result = await model_quality.evaluate_model_quality(
        settings_tmp,
        runtime_url="http://127.0.0.1:1",
        runtime_token=None,
        model_id="fixture-model",
        coding_root=settings_tmp.home / "coding",
        tool_worker=_FixtureToolWorker(),  # type: ignore[arg-type]
    )
    assert result["routing_accuracy"] == 1.0
    assert result["strict_json_first_pass_reliability"] == 1.0
    assert result["structured_json_reliability"] == 1.0
    assert result["coding_fixture_pass_rate"] == 1.0
    assert result["context_handling_reliability"] == 1.0
    assert result["routing"]["deterministic_router"]["included_in_model_accuracy"] is False
    assert max(result["context"]["context_token_counts"]) == 900
    blob = json.dumps(result)
    assert "ORCHID-731" not in blob
    assert "def clamp" not in blob


@pytest.mark.asyncio
async def test_benchmark_fixture_executes_only_in_bounded_tool_worker_root(
    settings_tmp: Any,
) -> None:
    project = settings_tmp.home / "fixture"
    project.mkdir()
    capability = secrets.token_urlsafe(32)
    executor = ToolWorkerExecutor(
        allowed_roots=(settings_tmp.home,),
        capability=capability,
    )
    response = await executor.execute(
        ToolWorkerRequest(
            request_id="benchmark-fixture",
            capability=capability,
            operation="benchmark_fixture",
            project_root=str(project),
            args={
                "fixture_files": {
                    "test_solution.py": (
                        "from solution import add\n\ndef test_add(): assert add(2, 3) == 5\n"
                    )
                },
                "candidate_file": "solution.py",
                "candidate_content": "def add(left, right):\n    return left + right\n",
                "expected_content": "def add(left, right):\n    return left + right\n",
                "test_argv": ["pytest", "-q"],
            },
            timeout_seconds=30,
            max_stdout_bytes=4_096,
            max_stderr_bytes=4_096,
        )
    )
    assert response.ok is True
    assert response.data["forbidden_file_modification"] is False
    assert response.data["syntax_or_compilation_failure"] is False
    assert response.data["unnecessary_change"] is False


def test_comparison_report_redaction_drops_raw_evidence() -> None:
    redacted = model_compare._redacted_benchmark(
        {
            "model_id": "candidate",
            "role": "brain",
            "model_basename": "candidate.gguf",
            "model_sha256": "a" * 64,
            "model_size": 4,
            "passed": True,
            "simulated": False,
            "prompt": "private prompt",
            "raw_output": "private answer",
            "absolute_path": "/private/repository/source.py",
            "quality": {"source": "secret"},
            "runs": [],
        }
    )
    blob = json.dumps(redacted)
    assert "private prompt" not in blob
    assert "private answer" not in blob
    assert "/private/" not in blob
    assert "secret" not in blob


def test_documented_durable_commands_match_registered_cli() -> None:
    runner = CliRunner()
    import_help = runner.invoke(app, ["april", "model", "import", "--help"])
    verify_help = runner.invoke(app, ["april", "model", "verify", "--help"])
    compare_help = runner.invoke(app, ["april", "model", "compare-setups", "--help"])
    reindex_help = runner.invoke(app, ["april", "memory", "reindex", "--help"])
    assert import_help.exit_code == 0
    assert "--sha256" in import_help.output
    assert "--wait" in import_help.output
    assert "--json" in import_help.output
    assert verify_help.exit_code == 0
    assert "--wait" in verify_help.output
    assert compare_help.exit_code == 0
    assert "--shared-model-id" in compare_help.output
    assert "--wait" in compare_help.output
    assert reindex_help.exit_code == 0
    assert "--wait" in reindex_help.output

    root = Path.cwd()
    documentation = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/background-jobs.md",
            "docs/macbookpro-acceptance.md",
            "docs/memory-design.md",
            "docs/production-readiness-roadmap.md",
            "docs/run-april-verification.md",
        )
    )
    assert "import-enqueue" not in documentation
    assert "model download --all-core --apply" not in documentation
    assert "run april model compare-setups" in documentation
    assert "run april memory reindex --wait" in documentation
