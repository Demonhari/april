from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from april_common.audit import AuditLogger
from services.api.server import create_app
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from services.permissions.tool_execution import ToolExecutionService
from skills.playbooks import PlaybookDefinition, PlaybookLoader, PlaybookMiner, PlaybookRunner
from skills.registry import ToolRegistry
from skills.schemas import ToolDefinition, ToolResult
from tests.test_core_api import auth, make_container


async def _echo(args: dict[str, object]) -> ToolResult:
    return ToolResult(
        ok=True,
        data={"echo": args},
        risk_level="read_only",
        permission_level=1,
    )


async def _dangerous(args: dict[str, object]) -> ToolResult:
    return ToolResult(
        ok=True,
        data={"dangerous": args},
        risk_level="code_write",
        permission_level=3,
    )


async def _tool_executor(settings_tmp) -> tuple[Database, SqliteMemory, ToolExecutionService]:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="test echo",
            permission_level=1,
            risk_level="read_only",
            allowed_agents={"playbook_agent"},
            executor=_echo,
        )
    )
    registry.register(
        ToolDefinition(
            name="dangerous",
            description="test dangerous",
            permission_level=3,
            risk_level="code_write",
            confirmation_required=True,
            allowed_agents={"playbook_agent"},
            executor=_dangerous,
        )
    )
    approvals = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60)
    return database, memory, ToolExecutionService(
        settings=settings_tmp,
        memory=memory,
        tool_registry=registry,
        permission_engine=PermissionEngine(registry),
        approvals=approvals,
    )


def _playbook(*tools: str) -> PlaybookDefinition:
    return PlaybookDefinition(
        id="test-playbook",
        name="Test playbook",
        agent_id="playbook_agent",
        status="active",
        trigger_examples=["run test playbook"],
        steps=[{"tool": tool, "args": {"value": tool}} for tool in tools],
    )


@pytest.mark.asyncio
async def test_playbook_runner_executes_safe_steps(settings_tmp) -> None:
    database, memory, executor = await _tool_executor(settings_tmp)
    try:
        result = await PlaybookRunner(executor, memory=memory).run(_playbook("echo"))
        assert result.status == "completed"
        assert result.steps_completed == 1
        rows = await database.fetchall("SELECT * FROM tool_calls WHERE tool = ?", ("echo",))
        assert len(rows) == 1
        # The run is persisted into playbook_runs.
        runs = await memory.list_playbook_runs(playbook_id="test-playbook")
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"
        assert runs[0]["steps_completed"] == 1
        assert runs[0]["completed_at"] is not None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_playbook_runner_pauses_for_l3_approval(settings_tmp) -> None:
    database, memory, executor = await _tool_executor(settings_tmp)
    try:
        result = await PlaybookRunner(executor, memory=memory).run(_playbook("dangerous"))
        assert result.status == "pending_approval"
        assert result.steps_completed == 0
        assert result.steps[0].approval is not None
        rows = await database.fetchall("SELECT * FROM approvals WHERE status = 'pending'")
        assert len(rows) == 1
        runs = await memory.list_playbook_runs(playbook_id="test-playbook")
        assert len(runs) == 1
        assert runs[0]["status"] == "pending_approval"
        assert "exact-action approval" in (runs[0]["detail"] or "")
    finally:
        await database.close()


def test_playbook_loader_ambiguous_trigger_falls_back(settings_tmp) -> None:
    loader = PlaybookLoader(settings_tmp.playbooks_path)
    first = _playbook("echo")
    second = first.model_copy(update={"id": "other-playbook", "name": "Other"})
    loader.adopt(first)
    loader.adopt(second)
    assert loader.match_trigger("please run test playbook") is None


def test_playbook_miner_uses_successful_tool_call_sequences() -> None:
    candidate = PlaybookMiner().mine(
        [
            {"tool": "search_files", "args": {"query": "TODO"}, "status": "executed"},
            {"tool": "read_file", "args": {"path": "README.md"}, "status": "executed"},
            {"tool": "write_file", "args": {"path": "x"}, "status": "failed"},
        ],
        trigger="inspect todos",
    )
    assert candidate is not None
    assert [step.tool for step in candidate.steps] == ["search_files", "read_file"]
    assert candidate.status == "candidate"


def test_playbook_loader_fuzzy_trigger_matches_close_message(settings_tmp) -> None:
    loader = PlaybookLoader(settings_tmp.playbooks_path)
    loader.adopt(_playbook("echo"))
    # Exact containment still matches; a small typo also routes.
    assert loader.match_trigger("please run test playbook now") is not None
    assert loader.match_trigger("run test playbok") is not None
    # Unrelated text never routes.
    assert loader.match_trigger("summarize my README") is None


def test_orchestrator_routes_unambiguous_trigger_to_playbook(settings_tmp) -> None:
    import anyio

    async def scenario() -> None:
        container = await make_container(settings_tmp)
        loader = PlaybookLoader(settings_tmp.playbooks_path)
        loader.adopt(
            PlaybookDefinition(
                id="reminder-playbook",
                name="Reminder playbook",
                agent_id="general_agent",
                status="active",
                trigger_examples=["run my reminder playbook"],
                steps=[{"tool": "create_reminder", "args": {"content": "stand up"}}],
            )
        )
        result = await container.orchestrator.chat("run my reminder playbook")
        assert result.status == "ok"
        assert "reminder-playbook" in result.final_message
        runs = await container.memory.list_playbook_runs(playbook_id="reminder-playbook")
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"
        reminders = await container.memory.list_reminders()
        assert reminders[0].content == "stand up"

    anyio.run(scenario)


def test_orchestrator_ambiguous_trigger_falls_back_to_brain(settings_tmp) -> None:
    import anyio

    async def scenario() -> None:
        container = await make_container(settings_tmp)
        loader = PlaybookLoader(settings_tmp.playbooks_path)
        base = PlaybookDefinition(
            id="one-playbook",
            name="One",
            agent_id="general_agent",
            status="active",
            trigger_examples=["plan my day"],
            steps=[{"tool": "create_reminder", "args": {"content": "one"}}],
        )
        loader.adopt(base)
        loader.adopt(
            base.model_copy(
                update={"id": "two-playbook", "name": "Two"}
            )
        )
        result = await container.orchestrator.chat("plan my day")
        # Ambiguity falls back to normal Brain routing (fake runtime answers).
        assert result.status == "ok"
        assert "playbook" not in result.final_message.casefold()
        assert await container.memory.list_playbook_runs() == []

    anyio.run(scenario)


def test_l3_playbook_adoption_requires_exact_action_approval(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    payload = {
        "id": "danger-playbook",
        "name": "Danger playbook",
        "status": "candidate",
        "agent_id": "coding_agent",
        "trigger_examples": ["apply the danger fix"],
        "steps": [{"tool": "run_command", "args": {"argv": ["pytest"]}}],
    }
    first = client.post("/playbooks/adopt", json=payload, headers=auth(settings_tmp))
    assert first.status_code == 200
    body = first.json()
    assert body["status"] == "pending_approval"
    assert body["required_permission_level"] >= 3
    approval_id = body["approval"]["approval_id"]
    # Not adopted yet: the playbook is not active on disk.
    loader = PlaybookLoader(settings_tmp.playbooks_path)
    assert loader.get("danger-playbook") is None

    # A modified definition must not consume the approval (exact-action bind).
    tampered = {**payload, "steps": [{"tool": "run_command", "args": {"argv": ["ruff"]}}]}
    rejected = client.post(
        f"/playbooks/adopt?approval_id={approval_id}",
        json=tampered,
        headers=auth(settings_tmp),
    )
    assert rejected.status_code in (400, 403)

    accepted = client.post(
        f"/playbooks/adopt?approval_id={approval_id}",
        json=payload,
        headers=auth(settings_tmp),
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "adopted"
    adopted = loader.get("danger-playbook")
    assert adopted is not None
    assert adopted.status == "active"


def test_safe_playbook_adoption_needs_no_approval(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    payload = {
        "id": "safe-playbook",
        "name": "Safe playbook",
        "status": "candidate",
        "agent_id": "general_agent",
        "trigger_examples": ["do the safe thing"],
        "steps": [{"tool": "create_reminder", "args": {"content": "safe"}}],
    }
    response = client.post("/playbooks/adopt", json=payload, headers=auth(settings_tmp))
    assert response.status_code == 200
    assert response.json()["status"] == "adopted"


def test_playbook_api_list_adopt_and_run(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    payload = {
        "id": "reminder-playbook",
        "name": "Reminder playbook",
        "status": "candidate",
        "agent_id": "general_agent",
        "trigger_examples": ["remind me"],
        "steps": [{"tool": "create_reminder", "args": {"content": "stand up"}}],
    }
    adopt = client.post("/playbooks/adopt", json=payload, headers=auth(settings_tmp))
    assert adopt.status_code == 200
    listed = client.get("/playbooks", headers=auth(settings_tmp))
    assert listed.status_code == 200
    assert listed.json()["playbooks"][0]["id"] == "reminder-playbook"

    run = client.post(
        "/playbooks/reminder-playbook/run",
        json={},
        headers=auth(settings_tmp),
    )
    assert run.status_code == 200
    assert run.json()["run"]["status"] == "completed"
    reminders = client.get("/reminders", headers=auth(settings_tmp)).json()["reminders"]
    assert reminders[0]["content"] == "stand up"
