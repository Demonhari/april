from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.runner.commands import finetune
from april_common.audit import AuditLogger
from services.api.server import create_app
from services.jobs.registry import default_job_registry
from services.jobs.store import JobStore, JobTransitionError
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest
from tests.test_core_api import auth, make_container
from tests.test_production_readiness_roadmap import _configured_home


def _configured_test_action(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    args = {"argv": ["pytest"], "repo_path": str(tmp_path)}
    payload = {"argv": ["pytest"], "cwd": str(tmp_path)}
    return args, payload


def _finetune_action(plan_id: str = "a" * 32) -> tuple[dict[str, Any], dict[str, Any]]:
    args = {
        "plan_id": plan_id,
        "dataset_sha256": "1" * 64,
        "configuration_sha256": "2" * 64,
        "base_model_sha256": "3" * 64,
        "trainer_sha256": "4" * 64,
        "evaluator_sha256": "5" * 64,
        "adapter_candidate_basename": f"{plan_id}.gguf",
    }
    return args, {"plan_id": plan_id}


def _model_import_action(tmp_path: Path) -> dict[str, Any]:
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


async def _create_approval(
    database: Database,
    tmp_path: Path,
    *,
    tool: str,
    args: dict[str, Any],
    permission_level: int,
    metadata: dict[str, Any] | None = None,
    expiry_seconds: int = 300,
) -> tuple[ApprovalStore, str]:
    approvals = ApprovalStore(
        database,
        AuditLogger(tmp_path / "audit.jsonl"),
        expiry_seconds=expiry_seconds,
    )
    response = await approvals.create(
        ApprovalRequest(
            tool=tool,
            args=args,
            agent="local-operator",
            permission_level=permission_level,
            risk_level="system_action" if permission_level == 4 else "code_write",
            affected_paths=[],
            expected_side_effects=["enqueue exact durable job"],
            metadata=metadata or {},
        ),
        actor="local-user",
        request_id="create-approved-job",
    )
    return approvals, response.approval_id


async def _approve(
    approvals: ApprovalStore,
    approval_id: str,
    *,
    tool: str,
    args: dict[str, Any],
) -> None:
    await approvals.approve_exact(
        approval_id=approval_id,
        tool=tool,
        args=args,
        actor="local-user",
        request_id="approve-approved-job",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job_type", "tool", "permission_level"),
    [
        ("configured_test", "test_runner", 3),
        ("finetune", "finetune", 4),
    ],
)
async def test_store_atomically_accepts_each_remaining_approved_job_type(
    tmp_path: Path,
    job_type: str,
    tool: str,
    permission_level: int,
) -> None:
    database = Database(tmp_path / f"{job_type}.db")
    await database.connect()
    await run_migrations(database)
    try:
        args, payload = (
            _configured_test_action(tmp_path)
            if job_type == "configured_test"
            else _finetune_action()
        )
        approvals, approval_id = await _create_approval(
            database,
            tmp_path,
            tool=tool,
            args=args,
            permission_level=permission_level,
        )
        await _approve(approvals, approval_id, tool=tool, args=args)
        store = JobStore(database, default_job_registry(finetune_enabled=True))
        job, created = await store.submit_with_exact_approval(
            job_type=job_type,
            payload=payload,
            owner="local-user",
            approval_id=approval_id,
            approval_tool=tool,
            approval_args=args,
        )
        replay, replay_created = await store.submit_with_exact_approval(
            job_type=job_type,
            payload=payload,
            owner="local-user",
            approval_id=approval_id,
            approval_tool=tool,
            approval_args=args,
        )
        assert created is True
        assert replay_created is False
        assert replay.id == job.id
        assert (await approvals.get(approval_id)).status == "consumed"
        assert len(await store.events(job.id)) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_concurrent_exact_approval_submissions_create_one_job(tmp_path: Path) -> None:
    database = Database(tmp_path / "concurrent.db")
    await database.connect()
    await run_migrations(database)
    try:
        args, payload = _configured_test_action(tmp_path)
        approvals, approval_id = await _create_approval(
            database,
            tmp_path,
            tool="test_runner",
            args=args,
            permission_level=3,
        )
        await _approve(approvals, approval_id, tool="test_runner", args=args)
        store = JobStore(database, default_job_registry())
        outcomes = await asyncio.gather(
            *(
                store.submit_with_exact_approval(
                    job_type="configured_test",
                    payload=payload,
                    owner="local-user",
                    approval_id=approval_id,
                    approval_tool="test_runner",
                    approval_args=args,
                )
                for _ in range(8)
            )
        )
        assert len({job.id for job, _created in outcomes}) == 1
        assert sum(created for _job, created in outcomes) == 1
        assert len(await store.list()) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_consumed_approval_rejects_changed_payload_action_and_scope(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "replay.db")
    await database.connect()
    await run_migrations(database)
    try:
        args, payload = _configured_test_action(tmp_path)
        approvals, approval_id = await _create_approval(
            database,
            tmp_path,
            tool="test_runner",
            args=args,
            permission_level=3,
        )
        await _approve(approvals, approval_id, tool="test_runner", args=args)
        store = JobStore(database, default_job_registry())
        memory = SqliteMemory(database)
        project_a = await memory.add_project(str(tmp_path / "project-a"), "project-a")
        project_b = await memory.add_project(str(tmp_path / "project-b"), "project-b")
        conversation_a = await memory.create_conversation(project_id=project_a.id)
        conversation_b = await memory.create_conversation(project_id=project_b.id)
        await store.submit_with_exact_approval(
            job_type="configured_test",
            payload=payload,
            owner="local-user",
            approval_id=approval_id,
            approval_tool="test_runner",
            approval_args=args,
            project_id=project_a.id,
            conversation_id=conversation_a,
        )
        with pytest.raises(JobTransitionError, match="approval_replay_mismatch"):
            await store.submit_with_exact_approval(
                job_type="configured_test",
                payload={**payload, "argv": ["pytest", "changed"]},
                owner="local-user",
                approval_id=approval_id,
                approval_tool="test_runner",
                approval_args=args,
                project_id=project_a.id,
                conversation_id=conversation_a,
            )
        with pytest.raises(JobTransitionError, match="approval_action_mismatch"):
            await store.submit_with_exact_approval(
                job_type="configured_test",
                payload=payload,
                owner="local-user",
                approval_id=approval_id,
                approval_tool="test_runner",
                approval_args={**args, "argv": ["pytest", "changed"]},
                project_id=project_a.id,
                conversation_id=conversation_a,
            )
        with pytest.raises(JobTransitionError, match="approval_replay_mismatch"):
            await store.submit_with_exact_approval(
                job_type="configured_test",
                payload=payload,
                owner="other-owner",
                approval_id=approval_id,
                approval_tool="test_runner",
                approval_args=args,
                project_id=project_b.id,
                conversation_id=conversation_b,
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_pending_denied_and_expired_approvals_cannot_enqueue(tmp_path: Path) -> None:
    database = Database(tmp_path / "states.db")
    await database.connect()
    await run_migrations(database)
    try:
        args, payload = _configured_test_action(tmp_path)
        store = JobStore(database, default_job_registry())

        pending_store, pending_id = await _create_approval(
            database,
            tmp_path,
            tool="test_runner",
            args=args,
            permission_level=3,
        )
        with pytest.raises(JobTransitionError, match="approval_not_approved"):
            await store.submit_with_exact_approval(
                job_type="configured_test",
                payload=payload,
                owner="local-user",
                approval_id=pending_id,
                approval_tool="test_runner",
                approval_args=args,
            )

        denied_store, denied_id = await _create_approval(
            database,
            tmp_path,
            tool="test_runner",
            args=args,
            permission_level=3,
        )
        await denied_store.deny(
            approval_id=denied_id,
            actor="local-user",
            request_id="deny-approved-job",
        )
        with pytest.raises(JobTransitionError, match="approval_not_approved"):
            await store.submit_with_exact_approval(
                job_type="configured_test",
                payload=payload,
                owner="local-user",
                approval_id=denied_id,
                approval_tool="test_runner",
                approval_args=args,
            )

        expired_store, expired_id = await _create_approval(
            database,
            tmp_path,
            tool="test_runner",
            args=args,
            permission_level=3,
        )
        await _approve(expired_store, expired_id, tool="test_runner", args=args)
        await database.execute(
            "UPDATE approvals SET expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00Z", expired_id),
        )
        with pytest.raises(JobTransitionError, match="approval_expired"):
            await store.submit_with_exact_approval(
                job_type="configured_test",
                payload=payload,
                owner="local-user",
                approval_id=expired_id,
                approval_tool="test_runner",
                approval_args=args,
            )
        assert await store.list() == []
        assert (await pending_store.get(pending_id)).status == "pending"
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["insert", "event", "consume"])
async def test_atomic_write_failures_roll_back_job_event_and_consumption(
    tmp_path: Path,
    failure_point: str,
) -> None:
    class FailingStore(JobStore):
        async def _insert_job_tx(self, *args: Any, **kwargs: Any) -> None:
            if failure_point == "insert":
                raise RuntimeError("injected_job_insert_failure")
            await super()._insert_job_tx(*args, **kwargs)

        async def _append_event_tx(self, *args: Any, **kwargs: Any) -> None:
            if failure_point == "event":
                raise RuntimeError("injected_initial_event_failure")
            await super()._append_event_tx(*args, **kwargs)

        async def _consume_approval_tx(self, *args: Any, **kwargs: Any) -> bool:
            if failure_point == "consume":
                raise RuntimeError("injected_approval_consumption_failure")
            return await super()._consume_approval_tx(*args, **kwargs)

    database = Database(tmp_path / f"{failure_point}.db")
    await database.connect()
    await run_migrations(database)
    try:
        args, payload = _configured_test_action(tmp_path)
        approvals, approval_id = await _create_approval(
            database,
            tmp_path,
            tool="test_runner",
            args=args,
            permission_level=3,
        )
        await _approve(approvals, approval_id, tool="test_runner", args=args)
        store = FailingStore(database, default_job_registry())
        expected_failure = {
            "insert": "injected_job_insert_failure",
            "event": "injected_initial_event_failure",
            "consume": "injected_approval_consumption_failure",
        }[failure_point]
        with pytest.raises(RuntimeError, match=expected_failure):
            await store.submit_with_exact_approval(
                job_type="configured_test",
                payload=payload,
                owner="local-user",
                approval_id=approval_id,
                approval_tool="test_runner",
                approval_args=args,
            )
        assert (await approvals.get(approval_id)).status == "approved"
        assert await store.list() == []
        events = await database.fetchall("SELECT * FROM background_job_events")
        assert events == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_cancellation_during_acceptance_rolls_back_and_releases_locks(
    tmp_path: Path,
) -> None:
    entered_event_write = asyncio.Event()
    never_complete = asyncio.Event()

    class BlockingStore(JobStore):
        async def _append_event_tx(self, *args: Any, **kwargs: Any) -> None:
            entered_event_write.set()
            await never_complete.wait()

    database = Database(tmp_path / "cancelled.db")
    await database.connect()
    await run_migrations(database)
    try:
        args, payload = _configured_test_action(tmp_path)
        approvals, approval_id = await _create_approval(
            database,
            tmp_path,
            tool="test_runner",
            args=args,
            permission_level=3,
        )
        await _approve(approvals, approval_id, tool="test_runner", args=args)
        blocked = BlockingStore(database, default_job_registry())
        submission = asyncio.create_task(
            blocked.submit_with_exact_approval(
                job_type="configured_test",
                payload=payload,
                owner="local-user",
                approval_id=approval_id,
                approval_tool="test_runner",
                approval_args=args,
            )
        )
        await asyncio.wait_for(entered_event_write.wait(), timeout=2)
        submission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submission
        assert (await approvals.get(approval_id)).status == "approved"
        assert await blocked.list() == []
        healthy = JobStore(database, default_job_registry())
        job = await asyncio.wait_for(
            healthy.submit(job_type="self_check", payload={}, owner="local-user"),
            timeout=2,
        )
        assert job.status.value == "queued"
    finally:
        await database.close()


async def _api_approval(
    container: Any,
    *,
    tool: str,
    args: dict[str, Any],
    permission_level: int,
    metadata: dict[str, Any] | None = None,
) -> str:
    response = await container.approvals.create(
        ApprovalRequest(
            tool=tool,
            args=args,
            agent="local-operator",
            permission_level=permission_level,
            risk_level="system_action" if permission_level == 4 else "code_write",
            metadata=metadata or {},
        ),
        actor="local-user",
        request_id="api-approved-job",
    )
    return response.approval_id


def test_api_uses_atomic_acceptance_for_all_approved_job_types(settings_tmp: Any) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    container.job_store = JobStore(
        container.database,
        default_job_registry(finetune_enabled=True),
    )
    client = TestClient(create_app(container))
    configured_args, configured_payload = _configured_test_action(settings_tmp.home)
    finetune_args, finetune_payload = _finetune_action()
    model_payload = _model_import_action(settings_tmp.home)
    cases = [
        ("configured_test", "test_runner", 3, configured_args, configured_payload),
        ("model_import", "model_import", 4, model_payload, model_payload),
        ("finetune", "finetune", 4, finetune_args, finetune_payload),
    ]
    job_ids: list[str] = []
    for job_type, tool, level, args, payload in cases:
        approval_id = anyio.run(
            lambda tool=tool, args=args, level=level: _api_approval(
                container,
                tool=tool,
                args=args,
                permission_level=level,
            )
        )
        response = client.post(
            "/jobs",
            headers=auth(settings_tmp),
            json={
                "job_type": job_type,
                "payload": payload,
                "approval_id": approval_id,
            },
        )
        assert response.status_code == 200
        job_ids.append(response.json()["id"])
        assert anyio.run(container.approvals.get, approval_id).status == "consumed"
        replay = client.post(
            "/jobs",
            headers=auth(settings_tmp),
            json={
                "job_type": job_type,
                "payload": payload,
                "approval_id": approval_id,
            },
        )
        assert replay.status_code == 200
        assert replay.json()["id"] == response.json()["id"]
    assert len(set(job_ids)) == 3
    audit_events = [
        json.loads(line)
        for line in settings_tmp.audit_path.read_text(encoding="utf-8").splitlines()
    ]
    consumed = [event for event in audit_events if event["event_type"] == "approval_consumed"]
    assert len(consumed) == 3


@pytest.mark.asyncio
async def test_api_concurrency_scope_replay_and_safe_jobs(settings_tmp: Any) -> None:
    container = await make_container(settings_tmp)
    app = create_app(container)
    transport = httpx.ASGITransport(app=app)
    args, payload = _configured_test_action(settings_tmp.home)
    approval_id = await _api_approval(
        container,
        tool="test_runner",
        args=args,
        permission_level=3,
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(
            *(
                client.post(
                    "/jobs",
                    headers=auth(settings_tmp),
                    json={
                        "job_type": "configured_test",
                        "payload": payload,
                        "approval_id": approval_id,
                    },
                )
                for _ in range(4)
            )
        )
        assert all(response.status_code == 200 for response in responses)
        assert len({response.json()["id"] for response in responses}) == 1
        assert container.job_store is not None
        assert len(await container.job_store.list()) == 1

        project = await container.memory.add_project(str(settings_tmp.home), "project")
        second_project = await container.memory.add_project(
            str(settings_tmp.home / "second-project"),
            "second-project",
        )
        conversation_a = await container.memory.create_conversation(
            "one",
            project_id=project.id,
        )
        conversation_b = await container.memory.create_conversation(
            "two",
            project_id=project.id,
        )
        conversation_c = await container.memory.create_conversation(
            "three",
            project_id=second_project.id,
        )
        scoped_id = await _api_approval(
            container,
            tool="test_runner",
            args=args,
            permission_level=3,
        )
        scoped = await client.post(
            "/jobs",
            headers=auth(settings_tmp),
            json={
                "job_type": "configured_test",
                "payload": payload,
                "approval_id": scoped_id,
                "project_id": project.id,
                "conversation_id": conversation_a,
            },
        )
        assert scoped.status_code == 200
        changed_scope = await client.post(
            "/jobs",
            headers=auth(settings_tmp),
            json={
                "job_type": "configured_test",
                "payload": payload,
                "approval_id": scoped_id,
                "project_id": project.id,
                "conversation_id": conversation_b,
            },
        )
        assert changed_scope.status_code == 409
        assert changed_scope.json()["detail"] == "approval_replay_mismatch"
        changed_project = await client.post(
            "/jobs",
            headers=auth(settings_tmp),
            json={
                "job_type": "configured_test",
                "payload": payload,
                "approval_id": scoped_id,
                "project_id": second_project.id,
                "conversation_id": conversation_c,
            },
        )
        assert changed_project.status_code == 409
        assert changed_project.json()["detail"] == "approval_replay_mismatch"

        safe = await client.post(
            "/jobs",
            headers=auth(settings_tmp),
            json={"job_type": "self_check", "payload": {}},
        )
        assert safe.status_code == 200


def test_api_conflicts_are_stable_and_do_not_leak_payloads(settings_tmp: Any) -> None:
    import anyio

    class ConsumptionRaceStore(JobStore):
        async def _consume_approval_tx(self, *args: Any, **kwargs: Any) -> bool:
            return False

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    secret = "PRIVATE-PAYLOAD-SENTINEL"
    payload = _model_import_action(settings_tmp.home)
    payload["source_path"] = str(settings_tmp.home / secret / "candidate.gguf")
    approval_id = anyio.run(
        lambda: _api_approval(
            container,
            tool="model_import",
            args=payload,
            permission_level=4,
        )
    )
    changed = {**payload, "destination": f"models/{secret}.gguf"}
    mismatch = client.post(
        "/jobs",
        headers=auth(settings_tmp),
        json={
            "job_type": "model_import",
            "payload": changed,
            "approval_id": approval_id,
        },
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"] == "approval_action_mismatch"
    assert secret not in mismatch.text
    assert str(settings_tmp.home) not in mismatch.text

    missing = client.post(
        "/jobs",
        headers=auth(settings_tmp),
        json={
            "job_type": "model_import",
            "payload": payload,
            "approval_id": "missing-approval",
        },
    )
    assert missing.status_code == 409
    assert missing.json()["detail"] == "approval_not_found"
    required = client.post(
        "/jobs",
        headers=auth(settings_tmp),
        json={"job_type": "model_import", "payload": payload},
    )
    assert required.status_code == 409
    assert required.json()["detail"] == "approval_required"

    denied_id = anyio.run(
        lambda: _api_approval(
            container,
            tool="model_import",
            args=payload,
            permission_level=4,
        )
    )
    anyio.run(
        lambda: container.approvals.deny(
            approval_id=denied_id,
            actor="local-user",
            request_id="api-deny",
        )
    )
    denied = client.post(
        "/jobs",
        headers=auth(settings_tmp),
        json={
            "job_type": "model_import",
            "payload": payload,
            "approval_id": denied_id,
        },
    )
    assert denied.status_code == 409
    assert denied.json()["detail"] == "approval_not_approved"

    expired_id = anyio.run(
        lambda: _api_approval(
            container,
            tool="model_import",
            args=payload,
            permission_level=4,
        )
    )
    anyio.run(
        container.database.execute,
        "UPDATE approvals SET expires_at = ? WHERE id = ?",
        ("2000-01-01T00:00:00Z", expired_id),
    )
    expired = client.post(
        "/jobs",
        headers=auth(settings_tmp),
        json={
            "job_type": "model_import",
            "payload": payload,
            "approval_id": expired_id,
        },
    )
    assert expired.status_code == 409
    assert expired.json()["detail"] == "approval_expired"

    race_id = anyio.run(
        lambda: _api_approval(
            container,
            tool="model_import",
            args=payload,
            permission_level=4,
        )
    )
    container.job_store = ConsumptionRaceStore(container.database, default_job_registry())
    race = client.post(
        "/jobs",
        headers=auth(settings_tmp),
        json={
            "job_type": "model_import",
            "payload": payload,
            "approval_id": race_id,
        },
    )
    assert race.status_code == 409
    assert race.json()["detail"] == "approval_consumption_race"
    assert anyio.run(container.approvals.get, race_id).status == "approved"
    assert anyio.run(container.job_store.list) == []

    unavailable = client.post(
        "/jobs",
        headers=auth(settings_tmp),
        json={"job_type": "finetune", "payload": {"plan_id": "a" * 32}},
    )
    assert unavailable.status_code == 409
    assert unavailable.json()["detail"] == "job_type_unavailable"


@pytest.mark.asyncio
async def test_finetune_cli_launch_is_atomic_idempotent_and_inactive(
    settings_tmp: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _configured_home(settings_tmp)
    source = settings.home / "reviewed-cli.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "type": "chat",
                    "prompt": f"prompt {index}",
                    "response": f"response {index}",
                }
            )
            + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(finetune, "load_settings", lambda: settings)
    plan, approval_id = await finetune._create_plan_and_approval(source, "base")
    first = await finetune._launch(plan.plan_id, approval_id)
    second = await finetune._launch(plan.plan_id, approval_id)
    assert second["job_id"] == first["job_id"]
    database = Database(settings.database_path)
    await database.connect()
    try:
        approvals = ApprovalStore(
            database,
            AuditLogger(settings.audit_path),
            expiry_seconds=settings.permissions.approval_expiry_seconds,
        )
        store = JobStore(database, default_job_registry(finetune_enabled=True))
        assert (await approvals.get(approval_id)).status == "consumed"
        assert len(await store.list()) == 1
    finally:
        await database.close()
    assert not (settings.evolution_path / "adapters" / "active" / "base.json").exists()


@pytest.mark.asyncio
async def test_finetune_cli_database_failure_cannot_leave_queued_job(
    settings_tmp: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingStore(JobStore):
        async def _append_event_tx(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("injected_cli_database_failure")

    settings = _configured_home(settings_tmp)
    source = settings.home / "reviewed-cli-failure.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "type": "chat",
                    "prompt": f"prompt {index}",
                    "response": f"response {index}",
                }
            )
            + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(finetune, "load_settings", lambda: settings)
    plan, approval_id = await finetune._create_plan_and_approval(source, "base")
    monkeypatch.setattr(finetune, "JobStore", FailingStore)
    with pytest.raises(RuntimeError, match="injected_cli_database_failure"):
        await finetune._launch(plan.plan_id, approval_id)
    database = Database(settings.database_path)
    await database.connect()
    try:
        approvals = ApprovalStore(
            database,
            AuditLogger(settings.audit_path),
            expiry_seconds=settings.permissions.approval_expiry_seconds,
        )
        store = JobStore(database, default_job_registry(finetune_enabled=True))
        assert (await approvals.get(approval_id)).status == "approved"
        assert await store.list() == []
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("dataset_sha256", "f" * 64),
        ("adapter_candidate_basename", "changed-candidate.gguf"),
    ],
)
async def test_finetune_cli_rejects_changed_plan_or_adapter_candidate(
    settings_tmp: Any,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    changed_value: str,
) -> None:
    settings = _configured_home(settings_tmp)
    source = settings.home / f"reviewed-cli-tamper-{field}.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "type": "chat",
                    "prompt": f"prompt {index}",
                    "response": f"response {index}",
                }
            )
            + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(finetune, "load_settings", lambda: settings)
    plan, approval_id = await finetune._create_plan_and_approval(source, "base")
    manifest_path = settings.evolution_path / "finetune" / "plans" / plan.plan_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = changed_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the immutable plan"):
        await finetune._launch(plan.plan_id, approval_id)
    database = Database(settings.database_path)
    await database.connect()
    try:
        approvals = ApprovalStore(
            database,
            AuditLogger(settings.audit_path),
            expiry_seconds=settings.permissions.approval_expiry_seconds,
        )
        store = JobStore(database, default_job_registry(finetune_enabled=True))
        assert (await approvals.get(approval_id)).status == "pending"
        assert await store.list() == []
    finally:
        await database.close()
