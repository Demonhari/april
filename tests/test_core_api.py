from __future__ import annotations

from fastapi.testclient import TestClient

from agents.registry import default_agent_registry
from april_common.audit import AuditLogger
from services.api.dependencies import ApiContainer
from services.api.server import create_app
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.retriever import MemoryRetriever
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from skills.registry import default_registry
from tests.conftest import FakeRuntimeClient


async def make_container(settings_tmp) -> ApiContainer:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    registry = default_registry()
    memory = SqliteMemory(database)
    vector_memory = VectorMemory(settings_tmp.vector_index_path)
    memory_retriever = MemoryRetriever(memory, vector_memory)
    runtime_client = FakeRuntimeClient()
    approvals = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60)
    from services.brain.orchestrator import AprilOrchestrator

    orchestrator = AprilOrchestrator(
        settings=settings_tmp,
        runtime_client=runtime_client,
        memory=memory,
        tool_registry=registry,
        permission_engine=PermissionEngine(registry),
        approvals=approvals,
        agent_registry=default_agent_registry(),
        memory_retriever=memory_retriever,
    )
    return ApiContainer(
        settings=settings_tmp,
        database=database,
        memory=memory,
        vector_memory=vector_memory,
        memory_retriever=memory_retriever,
        runtime_client=runtime_client,  # type: ignore[arg-type]
        tool_registry=registry,
        permission_engine=PermissionEngine(registry),
        approvals=approvals,
        agent_registry=default_agent_registry(),
        orchestrator=orchestrator,
    )


def auth(settings_tmp) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings_tmp.api.token}"}


def test_authentication(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 403
    response = client.get("/health")
    assert response.status_code == 200


def test_normal_chat_with_fake_backend(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat", json={"message": "April, plan my work today."}, headers=auth(settings_tmp)
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "ok"


def test_read_only_coding_analysis(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects",
        json={"path": str(settings_tmp.home)},
        headers=auth(settings_tmp),
    ).json()
    response = client.post(
        "/chat",
        json={
            "message": "April, check why the animation in this repository is broken.",
            "project_id": project["id"],
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "ok"


def test_repo_request_without_project_asks_for_selection(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "April, check why the animation in this repository is broken."},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "error"
    assert "project" in response.json()["result"]["final_message"].lower()


def test_approval_required_response(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/tools/request",
        json={
            "tool": "write_file",
            "agent": "coding_agent",
            "args": {"path": str(settings_tmp.home / "approved.txt"), "content": "ok"},
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "pending_approval"
    assert response.json()["approval"]["permission_level"] == 3


def test_approved_tool_executes_once_records_tool_call_and_audit(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    target = settings_tmp.home / "approved.txt"
    approval_response = client.post(
        "/tools/request",
        json={
            "tool": "write_file",
            "agent": "coding_agent",
            "args": {"path": str(target), "content": "approved"},
        },
        headers=auth(settings_tmp),
    )
    approval_id = approval_response.json()["approval"]["approval_id"]
    execute_response = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert execute_response.status_code == 200
    assert execute_response.json()["status"] == "executed"
    assert target.read_text(encoding="utf-8") == "approved"
    rows = anyio.run(container.database.fetchall, "SELECT * FROM tool_calls")
    assert len(rows) == 1
    assert "approved_tool_executed" in settings_tmp.audit_path.read_text(encoding="utf-8")
    replay = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert replay.status_code == 403


def test_failed_approved_execution_cannot_be_replayed(settings_tmp, tmp_path) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    outside = tmp_path.parent / "outside-approval.txt"
    approval_response = client.post(
        "/tools/request",
        json={
            "tool": "write_file",
            "agent": "coding_agent",
            "args": {"path": str(outside), "content": "denied"},
        },
        headers=auth(settings_tmp),
    )
    approval_id = approval_response.json()["approval"]["approval_id"]
    execute_response = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert execute_response.status_code == 200
    assert execute_response.json()["status"] == "failed"
    replay = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert replay.status_code == 403


def test_chat_stream_uses_runtime_stream(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "April, plan my work today."},
        headers=auth(settings_tmp),
    ) as response:
        body = response.read().decode()
    assert response.status_code == 200
    assert "event: token" in body
    assert body.index("event: token") < body.index("event: done")
    assert container.runtime_client.stream_called  # type: ignore[attr-defined]


def test_memory_retrieval_in_prompt(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)

    async def seed_memory() -> None:
        await container.memory.create_memory("I prefer deep work before meetings", reason="test")

    anyio.run(seed_memory)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat", json={"message": "April, plan my work today."}, headers=auth(settings_tmp)
    )
    assert response.status_code == 200
    prompt = "\n".join(
        message.content for message in container.runtime_client.last_messages  # type: ignore[attr-defined]
    )
    assert "Local APRIL memory" in prompt
    assert "deep work" in prompt


def test_vector_repo_chunks_return_citations(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    index_response = client.post(
        f"/projects/{project['id']}/index",
        json={},
        headers=auth(settings_tmp),
    )
    assert index_response.status_code == 200
    response = client.post(
        "/chat",
        json={
            "message": "April, check why the animation in this repository is broken.",
            "project_id": project["id"],
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    citations = response.json()["result"]["local_citations"]
    assert citations
    assert citations[0]["path"].endswith("README.md")


def test_request_size_limit(settings_tmp) -> None:
    import anyio

    small_api = settings_tmp.api.model_copy(update={"max_request_bytes": 10})
    limited_settings = settings_tmp.model_copy(update={"api": small_api})
    container = anyio.run(make_container, limited_settings)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "this request is too large"},
        headers=auth(limited_settings),
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_health_response(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database"]["ok"] is True


def test_project_add_normalizes_and_deduplicates(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    first = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    )
    second = client.post(
        "/projects",
        json={"path": str(settings_tmp.home / ".")},
        headers=auth(settings_tmp),
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert first.json()["path"] == str(settings_tmp.home)


def test_project_add_rejects_outside_allowed_roots(settings_tmp, tmp_path) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    outside = tmp_path.parent
    response = client.post(
        "/projects", json={"path": str(outside)}, headers=auth(settings_tmp)
    )
    assert response.status_code == 403
