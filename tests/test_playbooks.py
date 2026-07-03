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


async def _tool_executor(settings_tmp) -> tuple[Database, ToolExecutionService]:
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
    return database, ToolExecutionService(
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
    database, executor = await _tool_executor(settings_tmp)
    try:
        result = await PlaybookRunner(executor).run(_playbook("echo"))
        assert result.status == "completed"
        assert result.steps_completed == 1
        rows = await database.fetchall("SELECT * FROM tool_calls WHERE tool = ?", ("echo",))
        assert len(rows) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_playbook_runner_pauses_for_l3_approval(settings_tmp) -> None:
    database, executor = await _tool_executor(settings_tmp)
    try:
        result = await PlaybookRunner(executor).run(_playbook("dangerous"))
        assert result.status == "pending_approval"
        assert result.steps_completed == 0
        assert result.steps[0].approval is not None
        rows = await database.fetchall("SELECT * FROM approvals WHERE status = 'pending'")
        assert len(rows) == 1
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
