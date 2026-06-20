from __future__ import annotations

from fastapi.testclient import TestClient

from agents.registry import default_agent_registry
from april_common.audit import AuditLogger
from services.api.dependencies import ApiContainer
from services.api.server import create_app
from services.memory.database import Database
from services.memory.migrations import run_migrations
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
    approvals = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60)
    from services.brain.orchestrator import AprilOrchestrator

    orchestrator = AprilOrchestrator(
        settings=settings_tmp,
        runtime_client=FakeRuntimeClient(),
        memory=memory,
        tool_registry=registry,
        permission_engine=PermissionEngine(registry),
        approvals=approvals,
        agent_registry=default_agent_registry(),
    )
    return ApiContainer(
        settings=settings_tmp,
        database=database,
        memory=memory,
        vector_memory=VectorMemory(settings_tmp.vector_index_path),
        runtime_client=FakeRuntimeClient(),  # type: ignore[arg-type]
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
    response = client.post(
        "/chat",
        json={"message": "April, check why the animation in this repository is broken."},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "ok"


def test_approval_required_response(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "Apply the fix."},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "pending_approval"
    assert response.json()["result"]["pending_approval"]["permission_level"] == 3


def test_health_response(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database"]["ok"] is True
