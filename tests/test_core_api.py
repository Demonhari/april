from __future__ import annotations

import json
import subprocess

from fastapi.testclient import TestClient

from agents.registry import default_agent_registry
from april_common.audit import AuditLogger
from april_common.token_setup import generate_tokens
from services.api.dependencies import ApiContainer
from services.api.server import create_app
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.retriever import MemoryRetriever
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from services.permissions.schemas import ApprovalRequest
from services.permissions.tool_execution import ToolExecutionService
from skills.registry import default_registry
from tests.conftest import FakeRuntimeClient


async def make_container(
    settings_tmp, runtime_client: FakeRuntimeClient | None = None
) -> ApiContainer:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    registry = default_registry()
    memory = SqliteMemory(database)
    vector_memory = VectorMemory(settings_tmp.vector_index_path)
    memory_retriever = MemoryRetriever(memory, vector_memory)
    runtime_client = runtime_client or FakeRuntimeClient()
    approvals = ApprovalStore(
        database,
        AuditLogger(settings_tmp.audit_path),
        expiry_seconds=settings_tmp.permissions.approval_expiry_seconds,
    )
    permission_engine = PermissionEngine(registry)
    tool_executor = ToolExecutionService(
        settings=settings_tmp,
        memory=memory,
        tool_registry=registry,
        permission_engine=permission_engine,
        approvals=approvals,
    )
    from services.brain.orchestrator import AprilOrchestrator

    orchestrator = AprilOrchestrator(
        settings=settings_tmp,
        runtime_client=runtime_client,
        memory=memory,
        tool_registry=registry,
        permission_engine=permission_engine,
        approvals=approvals,
        tool_executor=tool_executor,
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
        permission_engine=permission_engine,
        approvals=approvals,
        tool_executor=tool_executor,
        agent_registry=default_agent_registry(),
        orchestrator=orchestrator,
    )


def auth(settings_tmp) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings_tmp.api.token}"}


def _settings_with_api_token(settings_tmp, token: str):
    return settings_tmp.model_copy(
        update={"api": settings_tmp.api.model_copy(update={"token": token})}
    )


def test_authentication(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 403
    response = client.get("/health")
    assert response.status_code == 200


def test_blank_configured_api_token_rejects_blank_bearer(settings_tmp) -> None:
    import anyio

    blank_settings = _settings_with_api_token(settings_tmp, "")
    container = anyio.run(make_container, blank_settings)
    client = TestClient(create_app(container))
    response = client.get("/readiness", headers={"Authorization": "Bearer "})
    assert response.status_code == 403
    assert response.json()["error"]["message"] == "API bearer token is not configured."


def test_blank_configured_api_token_rejects_missing_header(settings_tmp) -> None:
    import anyio

    blank_settings = _settings_with_api_token(settings_tmp, "")
    container = anyio.run(make_container, blank_settings)
    client = TestClient(create_app(container))
    response = client.get("/readiness")
    assert response.status_code == 403
    assert response.json()["error"]["message"] == "API bearer token is not configured."


def test_local_dev_token_authenticates_in_test(settings_tmp) -> None:
    import anyio

    local_settings = _settings_with_api_token(settings_tmp, "local-dev-token")
    container = anyio.run(make_container, local_settings)
    client = TestClient(create_app(container))
    response = client.get("/readiness", headers=auth(local_settings))
    assert response.status_code == 200


def test_readiness_reports_voice_loop_verdicts(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get("/readiness", headers=auth(settings_tmp))
    assert response.status_code == 200
    voice = response.json()["voice"]
    # The composite readiness verdicts are surfaced and are honest booleans.
    for key in (
        "openwakeword_available",
        "push_to_talk_ready",
        "wake_word_ready",
        "full_voice_loop_ready",
    ):
        assert isinstance(voice[key], bool)
    # With no whisper/piper/wake-word artifacts configured in the temp home, none of
    # the readiness rungs can be satisfied.
    assert voice["push_to_talk_ready"] is False
    assert voice["wake_word_ready"] is False
    assert voice["full_voice_loop_ready"] is False


def test_generated_api_token_authenticates(settings_tmp) -> None:
    import anyio

    token = generate_tokens().api_token
    strong_settings = _settings_with_api_token(settings_tmp, token)
    container = anyio.run(make_container, strong_settings)
    client = TestClient(create_app(container))
    response = client.get("/readiness", headers=auth(strong_settings))
    assert response.status_code == 200


def test_auth_error_does_not_include_token_values(settings_tmp) -> None:
    import anyio

    configured = "a-strong-local-api-token-value-123456"
    presented = "wrong-presented-token"
    strong_settings = _settings_with_api_token(settings_tmp, configured)
    container = anyio.run(make_container, strong_settings)
    client = TestClient(create_app(container))
    response = client.get("/readiness", headers={"Authorization": f"Bearer {presented}"})
    assert response.status_code == 403
    blob = json.dumps(response.json())
    assert configured not in blob
    assert presented not in blob


def test_lifespan_shutdown_closes_database(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    assert container.database.is_connected is True
    # Entering the TestClient context runs the FastAPI lifespan, whose shutdown
    # must release the database connection via ApiContainer.aclose().
    with TestClient(create_app(container)) as client:
        assert client.get("/health").status_code == 200
    assert container.database.is_connected is False


def test_normal_chat_with_fake_backend(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat", json={"message": "April, plan my work today."}, headers=auth(settings_tmp)
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "ok"
    assert response.json()["result"]["conversation_id"]


def test_conversation_id_reuses_recent_history(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    first = client.post(
        "/chat",
        json={"message": "April, plan my work today."},
        headers=auth(settings_tmp),
    ).json()
    conversation_id = first["result"]["conversation_id"]
    response = client.post(
        "/chat",
        json={"message": "Use that plan again.", "conversation_id": conversation_id},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    prompt = "\n".join(
        message.content
        for message in container.runtime_client.last_messages  # type: ignore[attr-defined]
    )
    assert "Recent conversation history" in prompt
    assert "April, plan my work today." in prompt


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
    rows = anyio.run(
        container.database.fetchall,
        "SELECT * FROM agent_runs WHERE agent = ? AND summary = ?",
        ("coding_agent", "structured agent loop"),
    )
    assert len(rows) == 1


def test_chat_routes_reading_agent_through_structured_loop(settings_tmp) -> None:
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
        json={"message": "Summarize README.md", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "ok"
    rows = anyio.run(
        container.database.fetchall,
        "SELECT * FROM agent_runs WHERE agent = ? AND summary = ?",
        ("reading_agent", "structured agent loop"),
    )
    assert len(rows) == 1


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


def test_chat_stream_specialist_uses_structured_events(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects",
        json={"path": str(settings_tmp.home)},
        headers=auth(settings_tmp),
    ).json()
    with client.stream(
        "POST",
        "/chat/stream",
        json={
            "message": "April, check why the animation in this repository is broken.",
            "project_id": project["id"],
        },
        headers=auth(settings_tmp),
    ) as response:
        body = response.read().decode()
    assert response.status_code == 200
    assert "event: routing" in body
    assert "event: agent_iteration" in body
    assert "event: final_answer" in body
    assert container.runtime_client.stream_called is False  # type: ignore[attr-defined]


def test_chat_stream_structured_approval_payload_shape(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects",
        json={"path": str(settings_tmp.home)},
        headers=auth(settings_tmp),
    ).json()
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    approval_events = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and '"event": "approval_required"' in line
    ]
    assert len(approval_events) == 1
    payload = approval_events[0]["payload"]
    assert set(payload) == {"approval", "message", "proposed_changes"}
    assert payload["approval"]["tool"] == "patch_applier"
    assert isinstance(payload["message"], str)
    assert isinstance(payload["proposed_changes"], list)


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
        message.content
        for message in container.runtime_client.last_messages  # type: ignore[attr-defined]
    )
    assert "Local APRIL memory" in prompt
    assert "deep work" in prompt


def test_system_action_agent_gets_no_prior_conversation_history(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    first = client.post(
        "/chat",
        json={"message": "April, plan my work today."},
        headers=auth(settings_tmp),
    ).json()
    conversation_id = first["result"]["conversation_id"]
    response = client.post(
        "/agents/run",
        json={
            "agent": "system_action_agent",
            "message": "Report current local action options.",
            "conversation_id": conversation_id,
            "options": {"structured": True},
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    prompt = "\n".join(
        message.content
        for message in container.runtime_client.last_messages  # type: ignore[attr-defined]
    )
    assert "Recent conversation history" not in prompt
    assert "April, plan my work today." not in prompt


def test_coding_agent_gets_project_chunks_only_for_selected_project(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()

    async def add_other_project():
        return await container.memory.add_project(str(settings_tmp.home / "other"))

    other_project = anyio.run(add_other_project)
    container.vector_memory.index_chunks(
        source_type="repo",
        source_id="selected",
        project_id=project["id"],
        chunks=[("selected.py", "selected project animation context", 1, 1)],
    )
    container.vector_memory.index_chunks(
        source_type="repo",
        source_id="other",
        project_id=other_project.id,
        chunks=[("other.py", "other project secret context", 1, 1)],
    )
    response = client.post(
        "/agents/run",
        json={
            "agent": "coding_agent",
            "message": "Inspect this repository animation context",
            "project_id": project["id"],
            "options": {"structured": True},
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    prompt = "\n".join(
        message.content
        for message in container.runtime_client.calls[0]  # type: ignore[attr-defined]
    )
    assert "selected project animation context" in prompt
    assert "other project secret context" not in prompt


def test_sensitive_memory_is_not_injected(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)

    async def seed_memory() -> None:
        await container.memory.create_memory("token should never be injected", reason="test")
        await container.memory.create_memory("I prefer morning planning", reason="test")

    anyio.run(seed_memory)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat", json={"message": "April, plan my work today."}, headers=auth(settings_tmp)
    )
    assert response.status_code == 200
    prompt = "\n".join(
        message.content
        for message in container.runtime_client.last_messages  # type: ignore[attr-defined]
    )
    assert "morning planning" in prompt
    assert "token should never be injected" not in prompt


def test_memory_create_search_export_delete_and_duplicate(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    payload = {
        "content": "I prefer concise answers",
        "memory_type": "preference",
        "reason": "explicit user request",
    }
    first = client.post("/memory", json=payload, headers=auth(settings_tmp))
    second = client.post("/memory", json=payload, headers=auth(settings_tmp))
    assert first.status_code == 200
    assert second.status_code == 200
    first_memory = first.json()["memory"]
    second_memory = second.json()["memory"]
    assert first_memory["id"] == second_memory["id"]
    assert first.json()["stored"] == "Stored preference memory."

    search = client.get(
        "/memory/search",
        params={"q": "concise"},
        headers=auth(settings_tmp),
    )
    assert search.status_code == 200
    assert [record["id"] for record in search.json()["results"]] == [first_memory["id"]]

    exported = client.get("/memory/export", headers=auth(settings_tmp)).json()["export"]
    assert "I prefer concise answers" in exported

    delete_response = client.delete(f"/memory/{first_memory['id']}", headers=auth(settings_tmp))
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert (
        client.get("/memory/search", params={"q": "*"}, headers=auth(settings_tmp)).json()[
            "results"
        ]
        == []
    )


def test_memory_write_requires_auth_and_valid_schema(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/memory",
        json={"content": "I prefer concise answers", "reason": "explicit"},
    )
    assert response.status_code == 403

    invalid = client.post(
        "/memory",
        json={"content": "x", "memory_type": "unsupported", "reason": "explicit"},
        headers=auth(settings_tmp),
    )
    assert invalid.status_code == 422


def test_non_explicit_preference_chat_is_not_written(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "I prefer concise answers."},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    memories = anyio.run(container.memory.list_memories)
    assert memories == []


def test_memory_rejects_secret_and_audit_omits_content(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    rejected = client.post(
        "/memory",
        json={
            "content": "api token abc123 should not be stored",
            "memory_type": "fact",
            "reason": "explicit",
        },
        headers=auth(settings_tmp),
    )
    assert rejected.status_code == 403

    response = client.post(
        "/memory",
        json={
            "content": "I prefer concise answers",
            "memory_type": "preference",
            "reason": "explicit",
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
    assert "memory_written" in audit_text
    assert "I prefer concise answers" not in audit_text
    assert "api token abc123" not in audit_text


def test_project_scoped_memory_search_and_export_are_isolated(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project_one = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()

    async def add_other_project():
        other = settings_tmp.home / "other-project"
        other.mkdir()
        return await container.memory.add_project(str(other))

    project_two = anyio.run(add_other_project)
    response = client.post(
        "/memory",
        json={
            "content": "Project alpha uses local SQLite",
            "memory_type": "project",
            "project_id": project_one["id"],
            "reason": "explicit",
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200

    isolated = client.get(
        "/memory/search",
        params={"q": "SQLite", "project_id": project_two.id},
        headers=auth(settings_tmp),
    )
    assert isolated.status_code == 200
    assert isolated.json()["results"] == []

    scoped = client.get(
        "/memory/search",
        params={"q": "SQLite", "project_id": project_one["id"]},
        headers=auth(settings_tmp),
    )
    assert len(scoped.json()["results"]) == 1
    exported = client.get(
        "/memory/export",
        params={"project_id": project_two.id},
        headers=auth(settings_tmp),
    ).json()["export"]
    assert "Project alpha" not in exported


def test_fake_backend_end_to_end_remember_request(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "Remember I prefer concise answers"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "ok"
    assert result["final_message"] == "Stored preference memory."
    memories = anyio.run(container.memory.search_memories, "concise")
    assert len(memories) == 1
    assert memories[0].content == "I prefer concise answers"
    rows = anyio.run(
        container.database.fetchall,
        "SELECT args_json FROM tool_calls WHERE tool = ?",
        ("remember_memory",),
    )
    assert rows
    assert "I prefer concise answers" not in rows[0]["args_json"]


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
    health = response.json()
    assert health["database"]["ok"] is True
    assert health["database"]["path"] == "[REDACTED]"
    assert health["vector_index"]["path"] == "[REDACTED]"
    assert str(settings_tmp.home) not in json.dumps(health)
    assert health["runtime"]["status"] == "ok"

    diagnostics = client.get("/diagnostics", headers=auth(settings_tmp))
    assert diagnostics.status_code == 200
    assert str(settings_tmp.database_path) in diagnostics.text


def test_health_degrades_when_runtime_unavailable(settings_tmp) -> None:
    import anyio

    class OfflineRuntime(FakeRuntimeClient):
        async def health(self, *, timeout: float | None = None) -> dict[str, object]:
            from april_common.errors import RuntimeUnavailableError

            raise RuntimeUnavailableError("offline")

    container = anyio.run(make_container, settings_tmp, OfflineRuntime())
    client = TestClient(create_app(container))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["runtime"]["status"] == "unavailable"


def test_task_plan_created_listed_and_completed(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat", json={"message": "April, plan my work today."}, headers=auth(settings_tmp)
    )
    assert response.status_code == 200
    tasks = client.get("/tasks", headers=auth(settings_tmp)).json()["tasks"]
    assert len(tasks) == 1
    task = tasks[0]
    assert task["intent"] == "planning"
    assert task["agent"] == "general_agent"
    assert task["status"] == "completed"
    assert task["steps"][0]["title"] == "Answer directly"


def test_task_plan_error_status(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "April, check why the animation in this repository is broken."},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    tasks = client.get("/tasks", headers=auth(settings_tmp)).json()["tasks"]
    assert tasks[0]["intent"] == "coding_repo_analysis"
    assert tasks[0]["status"] == "error"


def test_voice_input_preserves_conversation_id(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    first = client.post(
        "/chat",
        json={"message": "April, plan my work today."},
        headers=auth(settings_tmp),
    ).json()
    conversation_id = first["result"]["conversation_id"]
    voice = client.post(
        "/voice/input",
        json={"message": "Use that same plan.", "conversation_id": conversation_id},
        headers=auth(settings_tmp),
    )
    assert voice.status_code == 200
    assert voice.json()["result"]["conversation_id"] == conversation_id


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
    response = client.post("/projects", json={"path": str(outside)}, headers=auth(settings_tmp))
    assert response.status_code == 403


def test_code_modification_without_project_asks_for_selection(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "Apply the fix."},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "error"
    assert "project" in result["final_message"].lower()


def test_code_modification_creates_patch_and_pending_approval(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "pending_approval"
    assert result["pending_approval"]["tool"] == "patch_applier"
    patch_path = result["pending_approval"]["args"]["patch_path"]
    assert patch_path.startswith(str(settings_tmp.home / "data" / "artifacts" / "patches"))
    assert "Approval required" in result["final_message"]
    assert result["proposed_changes"][0]["path"] == "README.md"
    assert result["pending_approval"]["metadata"]["agent_run_id"]
    generated_patches = list(settings_tmp.home.joinpath("data/artifacts/patches").glob("*.patch"))
    assert len(generated_patches) == 1
    assert "fixed animation" in generated_patches[0].read_text(encoding="utf-8")
    rows = anyio.run(
        container.database.fetchall,
        "SELECT * FROM suspended_agent_runs WHERE approval_id = ?",
        (result["pending_approval"]["approval_id"],),
    )
    assert len(rows) == 1


def test_legacy_orchestrator_flag_keeps_one_shot_patch_path(settings_tmp, monkeypatch) -> None:
    import anyio

    monkeypatch.setenv("APRIL_LEGACY_ORCHESTRATOR", "1")
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "pending_approval"
    assert result["pending_approval"]["tool"] == "patch_applier"
    assert result["pending_approval"]["metadata"].get("agent_run_id") is None
    approve = client.post(
        "/tools/approve",
        json={"approval_id": result["pending_approval"]["approval_id"]},
        headers=auth(settings_tmp),
    )
    assert approve.status_code == 200
    assert approve.json()["status"] == "executed"


def test_direct_agent_run_validates_and_uses_structured_loop(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects",
        json={"path": str(settings_tmp.home)},
        headers=auth(settings_tmp),
    ).json()
    response = client.post(
        "/agents/run",
        json={
            "agent": "coding_agent",
            "message": "Check animation files",
            "project_id": project["id"],
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "ok"
    rows = anyio.run(
        container.database.fetchall,
        "SELECT * FROM agent_runs WHERE agent = ? AND summary = ?",
        ("coding_agent", "structured agent loop"),
    )
    assert len(rows) == 1


def test_direct_agent_run_rejects_unknown_agent(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/agents/run",
        json={"agent": "missing_agent", "message": "hello"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 403


def test_direct_agent_run_rejects_unstructured_option(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/agents/run",
        json={
            "agent": "general_agent",
            "message": "hello",
            "options": {"structured": False},
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 403


def test_direct_reasoning_agent_runs_on_brain_model(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/agents/run",
        json={"agent": "reasoning_agent", "message": "think"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "ok"
    rows = anyio.run(
        container.database.fetchall,
        "SELECT model_id FROM agent_runs WHERE agent = ? AND summary = ?",
        ("reasoning_agent", "structured agent loop"),
    )
    assert len(rows) == 1
    assert rows[0]["model_id"] == "april-brain"


def test_direct_coding_agent_requires_project(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/agents/run",
        json={"agent": "coding_agent", "message": "inspect"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 403


def test_direct_agent_run_suspends_and_resumes_after_approval(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects",
        json={"path": str(settings_tmp.home)},
        headers=auth(settings_tmp),
    ).json()
    response = client.post(
        "/agents/run",
        json={
            "agent": "coding_agent",
            "message": "Apply the fix.",
            "project_id": project["id"],
        },
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    approval_id = response.json()["result"]["pending_approval"]["approval_id"]
    approve = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert approve.status_code == 200
    assert approve.json()["status"] == "resumed"
    assert approve.json()["result"]["status"] == "ok"


def test_suspended_agent_denial_updates_run_without_execution(settings_tmp) -> None:
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
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    approval_id = response.json()["result"]["pending_approval"]["approval_id"]
    denied = client.post(
        "/tools/deny",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert denied.status_code == 200
    assert denied.json()["status"] == "denied"
    assert "fixed animation" not in (settings_tmp.home / "README.md").read_text(encoding="utf-8")
    rows = anyio.run(
        container.database.fetchall,
        "SELECT status FROM suspended_agent_runs WHERE approval_id = ?",
        (approval_id,),
    )
    assert rows[0]["status"] == "denied"


def test_suspended_agent_expired_approval_does_not_resume(settings_tmp) -> None:
    import anyio

    permissions = settings_tmp.permissions.model_copy(update={"approval_expiry_seconds": -1})
    expired_settings = settings_tmp.model_copy(update={"permissions": permissions})
    container = anyio.run(make_container, expired_settings)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects",
        json={"path": str(settings_tmp.home)},
        headers=auth(expired_settings),
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(expired_settings),
    )
    approval_id = response.json()["result"]["pending_approval"]["approval_id"]
    approve = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(expired_settings),
    )
    assert approve.status_code == 403
    assert "fixed animation" not in (settings_tmp.home / "README.md").read_text(encoding="utf-8")
    rows = anyio.run(
        container.database.fetchall,
        "SELECT status FROM suspended_agent_runs WHERE approval_id = ?",
        (approval_id,),
    )
    assert rows[0]["status"] == "expired"


def test_deleted_conversation_cannot_resume_suspended_agent(settings_tmp) -> None:
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
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    result = response.json()["result"]
    approval_id = result["pending_approval"]["approval_id"]
    conversation_id = result["conversation_id"]
    delete = client.delete(
        f"/conversations/{conversation_id}",
        headers=auth(settings_tmp),
    )
    assert delete.status_code == 200
    approve = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert approve.status_code == 403
    assert "fixed animation" not in (settings_tmp.home / "README.md").read_text(encoding="utf-8")


def test_missing_project_cannot_resume_suspended_agent(settings_tmp) -> None:
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
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    approval_id = response.json()["result"]["pending_approval"]["approval_id"]
    anyio.run(
        container.database.execute,
        "DELETE FROM projects WHERE id = ?",
        (project["id"],),
    )
    approve = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert approve.status_code == 403
    assert "fixed animation" not in (settings_tmp.home / "README.md").read_text(encoding="utf-8")


def test_suspended_agent_survives_service_restart(settings_tmp) -> None:
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
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    approval_id = response.json()["result"]["pending_approval"]["approval_id"]
    anyio.run(container.database.close)

    restarted = anyio.run(make_container, settings_tmp)
    restarted_client = TestClient(create_app(restarted))
    approve = restarted_client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert approve.status_code == 200
    assert approve.json()["status"] == "resumed"
    assert "fixed animation" in (settings_tmp.home / "README.md").read_text(encoding="utf-8")


def test_code_modification_approval_applies_exact_patch_once(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    approval_id = response.json()["result"]["pending_approval"]["approval_id"]
    approve_response = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "resumed"
    assert approve_response.json()["result"]["status"] == "ok"
    assert "fixed animation" in (settings_tmp.home / "README.md").read_text(encoding="utf-8")
    replay = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert replay.status_code == 403


def test_code_modification_external_project_uses_april_artifact_store(
    settings_tmp, tmp_path
) -> None:
    import anyio

    external_repo = tmp_path / "external-project"
    external_repo.mkdir()
    (external_repo / "README.md").write_text("# test repo\nanimation bug\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=external_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "april@example.local"],
        cwd=external_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "APRIL Test"],
        cwd=external_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "README.md"], cwd=external_repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=external_repo, check=True)
    paths = settings_tmp.paths.model_copy(
        update={"allowed_filesystem_roots": [settings_tmp.home, external_repo]}
    )
    scoped_settings = settings_tmp.model_copy(update={"paths": paths})
    container = anyio.run(make_container, scoped_settings)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects",
        json={"path": str(external_repo)},
        headers=auth(scoped_settings),
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(scoped_settings),
    )
    assert response.status_code == 200
    approval = response.json()["result"]["pending_approval"]
    patch_path = approval["args"]["patch_path"]
    assert patch_path.startswith(str(settings_tmp.home / "data" / "artifacts" / "patches"))
    assert not patch_path.startswith(str(external_repo))
    assert approval["metadata"]["project_id"] == project["id"]
    approve_response = client.post(
        "/tools/approve",
        json={"approval_id": approval["approval_id"]},
        headers=auth(scoped_settings),
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "resumed"
    assert approve_response.json()["result"]["status"] == "ok"
    assert "fixed animation" in (external_repo / "README.md").read_text(encoding="utf-8")


def test_patch_content_change_after_approval_is_rejected(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    approval = response.json()["result"]["pending_approval"]
    patch_path = settings_tmp.home.joinpath(approval["args"]["patch_path"])
    patch_path.write_text(
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,2 +1,3 @@\n"
        " # test repo\n"
        " animation bug\n"
        "+tampered\n",
        encoding="utf-8",
    )
    approve_response = client.post(
        "/tools/approve",
        json={"approval_id": approval["approval_id"]},
        headers=auth(settings_tmp),
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "failed"
    assert "tampered" not in (settings_tmp.home / "README.md").read_text(encoding="utf-8")
    replay = client.post(
        "/tools/approve",
        json={"approval_id": approval["approval_id"]},
        headers=auth(settings_tmp),
    )
    assert replay.status_code == 403


def test_patch_new_path_after_approval_is_rejected(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    approval = response.json()["result"]["pending_approval"]
    patch_path = settings_tmp.home.joinpath(approval["args"]["patch_path"])
    patch_path.write_text(
        patch_path.read_text(encoding="utf-8") + "\ndiff --git a/extra.txt b/extra.txt\n"
        "--- /dev/null\n"
        "+++ b/extra.txt\n"
        "@@ -0,0 +1 @@\n"
        "+extra\n",
        encoding="utf-8",
    )
    approve_response = client.post(
        "/tools/approve",
        json={"approval_id": approval["approval_id"]},
        headers=auth(settings_tmp),
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "failed"
    assert not (settings_tmp.home / "extra.txt").exists()


async def create_legacy_patch_approval(container: ApiContainer, patch_path: str):
    return await container.approvals.create(
        ApprovalRequest(
            tool="patch_applier",
            args={"repo_path": str(container.settings.home), "patch_path": patch_path},
            agent="coding_agent",
            permission_level=3,
            risk_level="code_write",
            affected_paths=[patch_path],
            expected_side_effects=["Apply legacy patch."],
            metadata={},
        ),
        actor="test",
        request_id="legacy-request",
    )


def test_legacy_patch_approval_without_artifact_metadata_is_rejected(settings_tmp) -> None:
    import anyio

    patch_path = settings_tmp.home / "legacy.patch"
    patch_path.write_text(
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,2 +1,3 @@\n"
        " # test repo\n"
        " animation bug\n"
        "+legacy apply\n",
        encoding="utf-8",
    )
    container = anyio.run(make_container, settings_tmp)
    approval = anyio.run(create_legacy_patch_approval, container, str(patch_path))
    client = TestClient(create_app(container))
    response = client.post(
        "/tools/approve",
        json={"approval_id": approval.approval_id},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "immutable artifact" in response.json()["result"]["stderr"]
    assert "legacy apply" not in (settings_tmp.home / "README.md").read_text(encoding="utf-8")
    replay = client.post(
        "/tools/approve",
        json={"approval_id": approval.approval_id},
        headers=auth(settings_tmp),
    )
    assert replay.status_code == 403


def test_code_modification_changed_args_cannot_reuse_approval(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    approval = response.json()["result"]["pending_approval"]
    changed_args = dict(approval["args"])
    changed_args["patch_path"] = str(settings_tmp.home / "other.patch")
    approve_response = client.post(
        "/tools/approve",
        json={
            "approval_id": approval["approval_id"],
            "tool": "patch_applier",
            "args": changed_args,
        },
        headers=auth(settings_tmp),
    )
    assert approve_response.status_code == 403


class UnsafePatchRuntimeClient(FakeRuntimeClient):
    def __init__(self, patch: str) -> None:
        super().__init__()
        self.patch = patch

    async def chat(self, **kwargs):
        response = await super().chat(**kwargs)
        joined = "\n".join(message.content for message in kwargs["messages"])
        lower = joined.lower()
        if (
            "return exactly one json object with type final_answer" in lower
            and "apply the fix" in lower
            and "tool result" not in lower
        ):
            return response.model_copy(
                update={
                    "content": (
                        '{"type":"tool_request","tool":"patch_generator","args":{'
                        f'"patch":{json.dumps(self.patch)}'
                        '},"reason":"Create an unsafe test patch."}'
                    )
                }
            )
        if "unified diff patch only" in joined.lower():
            return response.model_copy(update={"content": self.patch})
        return response


class PlannedOverrideRuntimeClient(FakeRuntimeClient):
    def __init__(self, decision_json: str) -> None:
        super().__init__()
        self.decision_json = decision_json

    async def chat(self, **kwargs):
        response = await super().chat(**kwargs)
        joined = "\n".join(message.content for message in kwargs["messages"])
        lower = joined.lower()
        if "return exactly one json object with type final_answer" in lower:
            path = self._extract_path_from_decision()
            if "override absolute path" in lower:
                return response.model_copy(
                    update={
                        "content": (
                            '{"type":"tool_request","tool":"read_file","args":{'
                            f'"path":{json.dumps(path)}'
                            '},"reason":"Attempt absolute read."}'
                        )
                    }
                )
            if "override repo path" in lower:
                return response.model_copy(
                    update={
                        "content": (
                            '{"type":"tool_request","tool":"search_files","args":{'
                            f'"path":{json.dumps(path)},'
                            '"query":"other-project-secret","limit":20'
                            '},"reason":"Attempt search override."}'
                        )
                    }
                )
        if "route this request" in lower or "route the user request" in lower:
            return response.model_copy(update={"content": self.decision_json})
        return response

    def _extract_path_from_decision(self) -> str:
        data = json.loads(self.decision_json)
        calls = data.get("planned_tool_calls") or []
        if not calls:
            return "."
        return str(calls[0].get("args", {}).get("path", "."))


def test_model_supplied_search_path_override_is_rejected(settings_tmp, tmp_path) -> None:
    import anyio

    other_project = settings_tmp.home.parent / "other-project-search"
    other_project.mkdir()
    (other_project / "secret.txt").write_text("other-project-secret", encoding="utf-8")
    runtime = PlannedOverrideRuntimeClient(
        '{"intent":"coding_repo_analysis","agent":"coding_agent","model_id":"april-coding",'
        '"tools_needed":[],"planned_tool_calls":[{"tool":"search_files","args":'
        f'{{"path":"{other_project}","query":"other-project-secret"}}'
        '}],"memory_queries":[],"permission_level":1,"risk_level":"read_only",'
        '"needs_confirmation":false,"task_steps":["Search"],"decision_summary":"Override attempt"}'
    )
    container = anyio.run(make_container, settings_tmp, runtime)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "override repo path", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 403
    assert "relative" in response.text.lower() or "project" in response.text.lower()


def test_model_supplied_absolute_read_path_outside_project_is_rejected(
    settings_tmp, tmp_path
) -> None:
    import anyio

    other_project = settings_tmp.home.parent / "other-project-read"
    other_project.mkdir()
    secret = other_project / "secret.txt"
    secret.write_text("other-project-secret", encoding="utf-8")
    runtime = PlannedOverrideRuntimeClient(
        '{"intent":"coding_repo_analysis","agent":"coding_agent","model_id":"april-coding",'
        '"tools_needed":[],"planned_tool_calls":[{"tool":"read_file","args":'
        f'{{"path":"{secret}"}}'
        '}],"memory_queries":[],"permission_level":1,"risk_level":"read_only",'
        '"needs_confirmation":false,"task_steps":["Read"],"decision_summary":"Override attempt"}'
    )
    container = anyio.run(make_container, settings_tmp, runtime)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "override absolute path", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 403


def test_code_modification_rejects_patch_outside_project(settings_tmp) -> None:
    import anyio

    runtime = UnsafePatchRuntimeClient(
        "diff --git a/../outside.txt b/../outside.txt\n"
        "--- a/../outside.txt\n"
        "+++ b/../outside.txt\n"
        "@@ -0,0 +1 @@\n"
        "+outside\n"
    )
    container = anyio.run(make_container, settings_tmp, runtime)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 403
    assert "project" in response.text.lower() or "patch" in response.text.lower()


def test_code_modification_rejects_patch_touching_git(settings_tmp) -> None:
    import anyio

    runtime = UnsafePatchRuntimeClient(
        "diff --git a/.git/config b/.git/config\n"
        "--- a/.git/config\n"
        "+++ b/.git/config\n"
        "@@ -0,0 +1 @@\n"
        "+bad\n"
    )
    container = anyio.run(make_container, settings_tmp, runtime)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 403
    assert ".git" in response.text or "patch" in response.text.lower()


def test_code_modification_rejects_symlink_escape(settings_tmp, tmp_path) -> None:
    import anyio

    outside = settings_tmp.home.parent / "outside-readme.md"
    outside.write_text("# outside\nanimation bug\n", encoding="utf-8")
    readme = settings_tmp.home / "README.md"
    readme.unlink()
    readme.symlink_to(outside)
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 403
    assert "patch" in response.text.lower() or "project" in response.text.lower()


def test_git_commit_staged_change_after_approval_is_rejected(settings_tmp) -> None:
    import anyio

    subprocess.run(["git", "init"], cwd=settings_tmp.home, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "april@example.local"],
        cwd=settings_tmp.home,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "APRIL Test"],
        cwd=settings_tmp.home,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "README.md"], cwd=settings_tmp.home, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=settings_tmp.home, check=True)
    (settings_tmp.home / "README.md").write_text(
        "# test repo\nanimation bug\nfirst staged\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "README.md"], cwd=settings_tmp.home, check=True)

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    approval_response = client.post(
        "/tools/request",
        json={
            "tool": "git_commit",
            "agent": "coding_agent",
            "args": {
                "repo_path": str(settings_tmp.home),
                "project_id": project["id"],
                "message": "approved",
            },
        },
        headers=auth(settings_tmp),
    )
    assert approval_response.status_code == 200
    approval_id = approval_response.json()["approval"]["approval_id"]
    (settings_tmp.home / "README.md").write_text(
        "# test repo\nanimation bug\nchanged staged\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "README.md"], cwd=settings_tmp.home, check=True)
    execute_response = client.post(
        "/tools/approve",
        json={"approval_id": approval_id},
        headers=auth(settings_tmp),
    )
    assert execute_response.status_code == 200
    assert execute_response.json()["status"] == "failed"


def test_direct_tool_request_requires_registered_project_for_repo_tools(
    settings_tmp, tmp_path
) -> None:
    import anyio

    other_repo = tmp_path / "unregistered"
    other_repo.mkdir()
    subprocess.run(["git", "init"], cwd=other_repo, check=True, capture_output=True)
    paths = settings_tmp.paths.model_copy(
        update={"allowed_filesystem_roots": [settings_tmp.home, other_repo]}
    )
    scoped_settings = settings_tmp.model_copy(update={"paths": paths})
    container = anyio.run(make_container, scoped_settings)
    client = TestClient(create_app(container))
    response = client.post(
        "/tools/request",
        json={
            "tool": "git_status",
            "agent": "coding_agent",
            "args": {"repo_path": str(other_repo)},
        },
        headers=auth(scoped_settings),
    )
    assert response.status_code == 403
    assert "registered selected project" in response.text


def test_conversation_cannot_switch_project(settings_tmp, tmp_path) -> None:
    import anyio

    other_project = tmp_path / "other-project"
    other_project.mkdir()
    (other_project / "README.md").write_text("# other\n", encoding="utf-8")
    paths = settings_tmp.paths.model_copy(
        update={"allowed_filesystem_roots": [settings_tmp.home, other_project]}
    )
    scoped_settings = settings_tmp.model_copy(update={"paths": paths})
    container = anyio.run(make_container, scoped_settings)
    client = TestClient(create_app(container))
    first_project = client.post(
        "/projects",
        json={"path": str(settings_tmp.home)},
        headers=auth(scoped_settings),
    ).json()
    second_project = client.post(
        "/projects",
        json={"path": str(other_project)},
        headers=auth(scoped_settings),
    ).json()
    first = client.post(
        "/chat",
        json={"message": "April, plan my work today.", "project_id": first_project["id"]},
        headers=auth(scoped_settings),
    ).json()
    response = client.post(
        "/chat",
        json={
            "message": "Use the same conversation elsewhere.",
            "conversation_id": first["result"]["conversation_id"],
            "project_id": second_project["id"],
        },
        headers=auth(scoped_settings),
    )
    assert response.status_code == 403
    assert "project scope cannot change" in response.text


def test_run_command_cwd_is_forced_to_selected_project(settings_tmp, tmp_path) -> None:
    import anyio

    other_project = tmp_path / "other-command"
    other_project.mkdir()
    paths = settings_tmp.paths.model_copy(
        update={"allowed_filesystem_roots": [settings_tmp.home, other_project]}
    )
    scoped_settings = settings_tmp.model_copy(update={"paths": paths})
    container = anyio.run(make_container, scoped_settings)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects",
        json={"path": str(settings_tmp.home)},
        headers=auth(scoped_settings),
    ).json()
    response = client.post(
        "/tools/request",
        json={
            "tool": "run_command",
            "agent": "coding_agent",
            "args": {
                "project_id": project["id"],
                "argv": ["pytest"],
                "cwd": str(other_project),
            },
        },
        headers=auth(scoped_settings),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "pending_approval"
    assert response.json()["approval"]["args"]["cwd"] == str(settings_tmp.home)


def test_project_index_records_audit_and_tool_call(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    response = client.post(
        f"/projects/{project['id']}/index",
        json={},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    rows = anyio.run(
        container.database.fetchall, "SELECT * FROM tool_calls WHERE tool = ?", ("repo_indexer",)
    )
    assert len(rows) == 1
    assert "tool_executed" in settings_tmp.audit_path.read_text(encoding="utf-8")


def test_project_scoped_read_cannot_access_another_project(settings_tmp, tmp_path) -> None:
    import anyio

    other_project = tmp_path / "other-read"
    other_project.mkdir()
    (other_project / "secret.txt").write_text("other secret", encoding="utf-8")
    paths = settings_tmp.paths.model_copy(
        update={"allowed_filesystem_roots": [settings_tmp.home, other_project]}
    )
    scoped_settings = settings_tmp.model_copy(update={"paths": paths})
    container = anyio.run(make_container, scoped_settings)
    client = TestClient(create_app(container))
    project = client.post(
        "/projects",
        json={"path": str(settings_tmp.home)},
        headers=auth(scoped_settings),
    ).json()
    response = client.post(
        "/tools/request",
        json={
            "tool": "read_file",
            "agent": "coding_agent",
            "args": {"project_id": project["id"], "path": "../other-read/secret.txt"},
        },
        headers=auth(scoped_settings),
    )
    assert response.status_code == 403


def test_recorded_permission_uses_policy_not_executor_output(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=auth(settings_tmp)
    ).json()
    index = client.post(
        f"/projects/{response['id']}/index",
        json={},
        headers=auth(settings_tmp),
    )
    assert index.status_code == 200
    rows = anyio.run(
        container.database.fetchall, "SELECT * FROM tool_calls WHERE tool = ?", ("repo_indexer",)
    )
    assert rows[0]["permission_level"] == 2


def test_disabled_external_action_is_denied(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/tools/request",
        json={"tool": "git_push", "agent": "system_action_agent", "args": {}},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 403
    assert "External actions are disabled" in response.text


def test_reminders_and_tasks_api_are_authenticated(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    unauthenticated = client.get("/reminders")
    assert unauthenticated.status_code == 403
    created = client.post(
        "/reminders",
        json={"content": "stand up", "due_at": "2026-06-21T09:00:00Z"},
        headers=auth(settings_tmp),
    )
    assert created.status_code == 200
    reminder_id = created.json()["reminder"]["id"]
    listed = client.get("/reminders", headers=auth(settings_tmp))
    assert listed.json()["reminders"][0]["content"] == "stand up"
    tasks = client.get("/tasks", headers=auth(settings_tmp))
    assert tasks.status_code == 200
    assert tasks.json()["tasks"] == []
    deleted = client.delete(f"/reminders/{reminder_id}", headers=auth(settings_tmp))
    assert deleted.json() == {"deleted": True}


def test_cors_setting_is_applied(settings_tmp) -> None:
    import anyio

    api = settings_tmp.api.model_copy(update={"cors_enabled": True})
    cors_settings = settings_tmp.model_copy(update={"api": api})
    container = anyio.run(make_container, cors_settings)
    client = TestClient(create_app(container))
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://127.0.0.1",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1"
