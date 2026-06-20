from __future__ import annotations

import subprocess

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
    approvals = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60)
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
        message.content
        for message in container.runtime_client.last_messages  # type: ignore[attr-defined]
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
    assert "README.md" in result["final_message"]
    assert result["proposed_changes"][0]["path"] == "README.md"
    generated_patches = list(settings_tmp.home.joinpath("data/artifacts/patches").glob("*.patch"))
    assert len(generated_patches) == 1
    assert "fixed animation" in generated_patches[0].read_text(encoding="utf-8")


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
    assert approve_response.json()["status"] == "executed"
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
    assert approve_response.json()["status"] == "executed"
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
        if "route this request" in joined.lower():
            return response.model_copy(update={"content": self.decision_json})
        return response


def test_model_supplied_search_path_is_overridden_by_selected_project(
    settings_tmp, tmp_path
) -> None:
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
    assert response.status_code == 200
    prompt = "\n".join(message.content for message in runtime.last_messages)
    assert "other-project-secret" not in prompt


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
    result = response.json()["result"]
    assert result["status"] == "error"
    assert "safe patch" in result["final_message"].lower()


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
    result = response.json()["result"]
    assert result["status"] == "error"
    assert "safe patch" in result["final_message"].lower()


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
    result = response.json()["result"]
    assert result["status"] == "error"
    assert "safe patch" in result["final_message"].lower()


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
