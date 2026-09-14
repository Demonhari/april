from __future__ import annotations

import json
from typing import Any

import pytest

from agents.coding.agent import coding_agent
from agents.reasoning.agent import reasoning_agent
from april_common.audit import AuditLogger
from services.april_runtime.schemas import ChatMessage, ChatResponse, Usage
from services.brain.agent_loop import StructuredAgentLoop
from services.brain.task_contract import CapabilityCeiling, TaskContract, VerificationRequirements
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from services.permissions.schemas import PermissionDecision
from services.permissions.tool_execution import ToolExecutionOutcome, ToolExecutionService
from skills.registry import default_registry
from skills.schemas import ToolResult


class SequenceRuntimeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[ChatMessage]] = []
        self.response_formats: list[Any] = []

    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: Any | None = None,
        response_format: Any | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        self.calls.append(messages)
        self.response_formats.append(response_format)
        content = self.responses.pop(0)
        return ChatResponse(
            request_id=request_id or "request",
            model_id=model_id,
            content=content,
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


async def make_loop(settings_tmp, responses: list[str]):
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    registry = default_registry()
    permission_engine = PermissionEngine(registry)
    approvals = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60)
    executor = ToolExecutionService(
        settings=settings_tmp,
        memory=memory,
        tool_registry=registry,
        permission_engine=permission_engine,
        approvals=approvals,
    )
    runtime = SequenceRuntimeClient(responses)
    loop = StructuredAgentLoop(runtime_client=runtime, tool_executor=executor, memory=memory)  # type: ignore[arg-type]
    project = await memory.add_project(str(settings_tmp.home))
    conversation_id = await memory.create_conversation(project_id=project.id)
    context = await executor.context(
        request_id="request",
        conversation_id=conversation_id,
        actor="local-user",
        agent_id="coding_agent",
        project_id=project.id,
        source="orchestrator",
    )
    return loop, context, memory, database, runtime


@pytest.mark.asyncio
async def test_structured_agent_loop_allowed_tool_executes(settings_tmp) -> None:
    loop, context, _memory, database, runtime = await make_loop(
        settings_tmp,
        [
            '{"type":"tool_request","tool":"read_file","args":{"path":"README.md"},'
            '"reason":"Inspect file"}',
            '{"type":"final_answer","message":"Read the file.","summary":"done","citations":[]}',
        ],
    )
    result = await loop.run(
        agent=coding_agent(),
        message="read",
        context=context,
        request_id="request",
    )
    assert result.status == "ok"
    rows = await database.fetchall("SELECT * FROM tool_calls WHERE tool = ?", ("read_file",))
    assert len(rows) == 1
    iterations = await database.fetchall("SELECT * FROM agent_iterations")
    assert len(iterations) >= 2
    assert [message.role for message in runtime.calls[1][-2:]] == ["assistant", "tool"]
    request = json.loads(runtime.calls[1][-2].content)
    result_payload = json.loads(runtime.calls[1][-1].content)
    assert request["type"] == "tool_request"
    assert result_payload["tool"] == "read_file"
    await database.close()


@pytest.mark.asyncio
async def test_structured_agent_loop_blocked_tool_denied(settings_tmp) -> None:
    loop, context, _memory, database, _runtime = await make_loop(
        settings_tmp,
        ['{"type":"tool_request","tool":"git_push","args":{},"reason":"bad"}'],
    )
    result = await loop.run(
        agent=coding_agent(),
        message="push",
        context=context,
        request_id="request",
    )
    assert result.status == "error"
    assert "disallowed tool" in result.final_message
    await database.close()


@pytest.mark.asyncio
async def test_structured_agent_loop_repairs_malformed_json(settings_tmp) -> None:
    loop, context, _memory, database, runtime = await make_loop(
        settings_tmp,
        [
            "not json",
            '{"type":"final_answer","message":"repaired","summary":"ok","citations":[]}',
        ],
    )
    result = await loop.run(
        agent=coding_agent(),
        message="repair",
        context=context,
        request_id="request",
    )
    assert result.status == "ok"
    assert result.final_message == "repaired"
    assert len(runtime.calls) == 2
    await database.close()


@pytest.mark.asyncio
async def test_structured_agent_loop_failed_repair_returns_error(settings_tmp) -> None:
    loop, context, _memory, database, _runtime = await make_loop(
        settings_tmp,
        ["not json", "still not json"],
    )
    result = await loop.run(
        agent=coding_agent(),
        message="repair",
        context=context,
        request_id="request",
    )
    assert result.status == "error"
    assert "malformed structured output" in result.final_message
    await database.close()


@pytest.mark.asyncio
async def test_structured_agent_loop_rejects_empty_final_answer(settings_tmp) -> None:
    loop, context, _memory, database, _runtime = await make_loop(
        settings_tmp,
        ['{"type":"final_answer","message":"","summary":"empty","citations":[]}'],
    )
    result = await loop.run(
        agent=reasoning_agent(),
        message="empty",
        context=context,
        request_id="request",
    )
    assert result.status == "error"
    assert "complete user-facing answer" in result.final_message
    await database.close()


@pytest.mark.asyncio
async def test_structured_agent_loop_level_three_suspends(settings_tmp) -> None:
    loop, context, _memory, database, _runtime = await make_loop(
        settings_tmp,
        [
            '{"type":"tool_request","tool":"write_file",'
            '"args":{"path":"generated.txt","content":"ok"},"reason":"write"}'
        ],
    )
    result = await loop.run(
        agent=coding_agent(),
        message="write",
        context=context,
        request_id="request",
    )
    assert result.status == "pending_approval"
    assert result.pending_approval is not None
    rows = await database.fetchall("SELECT * FROM approvals WHERE status = 'pending'")
    assert len(rows) == 1
    suspended = await database.fetchone("SELECT messages_json FROM suspended_agent_runs")
    assert suspended is not None
    messages = json.loads(suspended["messages_json"])
    assert messages[-1]["role"] == "assistant"
    assert json.loads(messages[-1]["content"])["type"] == "tool_request"
    assert all(message["role"] != "tool" for message in messages)
    await database.close()


@pytest.mark.asyncio
async def test_structured_agent_loop_iteration_limit_is_enforced(settings_tmp) -> None:
    repeated_request = (
        '{"type":"tool_request","tool":"read_file","args":{"path":"README.md"},'
        '"reason":"keep reading"}'
    )
    loop, context, _memory, database, _runtime = await make_loop(
        settings_tmp,
        [repeated_request] * 5,
    )
    result = await loop.run(
        agent=coding_agent(),
        message="loop",
        context=context,
        request_id="request",
    )
    assert result.status == "error"
    assert "iteration limit" in result.final_message
    rows = await database.fetchall("SELECT * FROM agent_runs WHERE status = 'error'")
    assert len(rows) == 1
    await database.close()


@pytest.mark.asyncio
async def test_agent_without_model_unavailable(settings_tmp) -> None:
    from agents.base import BaseAgent

    loop, context, _memory, database, _runtime = await make_loop(settings_tmp, [])
    base = reasoning_agent()
    model_less = BaseAgent(base.config.model_copy(update={"model_id": None}))
    result = await loop.run(
        agent=model_less,
        message="think deeply",
        context=context,
        request_id="request",
    )
    assert result.status == "unavailable"
    await database.close()


def _verified_contract(context, settings_tmp) -> TaskContract:
    return TaskContract(
        run_id="request",
        user_goal="repair the repository",
        project_id=context.project_id,
        project_root=str(settings_tmp.home),
        task_type="verified_code_modification",
        risk_class="write",
        verification=VerificationRequirements(required=True),
        maximum_replan_attempts=1,
        capability_ceiling=CapabilityCeiling(
            allowed_tools=frozenset(coding_agent().config.allowed_tools),
            maximum_risk_level=3,
            project_id=context.project_id,
            project_root=str(settings_tmp.home),
        ),
    )


class _VerificationExecutor:
    def __init__(self, result: ToolResult) -> None:
        self.result = result

    async def request_or_execute(self, *, tool, args, context, **kwargs):
        del tool, context, kwargs
        return ToolExecutionOutcome(
            status="executed",
            args=args,
            permission=PermissionDecision(
                allowed=True,
                permission_level=3,
                risk_level="code_write",
                confirmation_required=False,
                reason="test",
            ),
            result=self.result,
        )


@pytest.mark.asyncio
async def test_verified_coding_final_answer_requires_machine_verification(settings_tmp) -> None:
    loop, context, _memory, database, _runtime = await make_loop(
        settings_tmp,
        [
            '{"type":"final_answer","message":"done","summary":"done","citations":[]}',
            '{"type":"final_answer","message":"still done","summary":"done","citations":[]}',
        ],
    )
    result = await loop.run(
        agent=coding_agent(),
        message="repair",
        context=context,
        request_id="request",
        task_contract=_verified_contract(context, settings_tmp),
    )
    assert result.status == "error"
    assert "incomplete" in result.final_message
    await database.close()


@pytest.mark.asyncio
async def test_verified_coding_final_answer_accepts_current_passing_test(settings_tmp) -> None:
    loop, context, memory, database, _runtime = await make_loop(
        settings_tmp,
        [
            '{"type":"tool_request","tool":"test_runner","args":{"argv":["pytest"]},"reason":"verify"}',
            '{"type":"final_answer","message":"verified","summary":"done","citations":[]}',
        ],
    )
    loop.tool_executor = _VerificationExecutor(
        ToolResult(
            ok=True,
            stdout="1 passed",
            data={"returncode": 0},
            risk_level="code_write",
            permission_level=3,
        )
    )  # type: ignore[assignment]
    result = await loop.run(
        agent=coding_agent(),
        message="repair",
        context=context,
        request_id="request",
        task_contract=_verified_contract(context, settings_tmp),
    )
    assert result.status == "ok"
    assert result.final_message == "verified"
    run = await memory.database.fetchone(
        "SELECT metadata_json FROM agent_runs ORDER BY created_at DESC LIMIT 1"
    )
    assert run is not None
    assert "run_control" in json.loads(str(run["metadata_json"]))
    await database.close()


@pytest.mark.asyncio
async def test_structured_agent_loop_rejects_invalid_tool_request(settings_tmp) -> None:
    # A tool_request carrying an extra forbidden field is schema-invalid: it must
    # be repaired (never executed as-is). Here the repair resolves to a plain
    # answer, so NO tool call is recorded for the malformed request.
    loop, context, _memory, database, runtime = await make_loop(
        settings_tmp,
        [
            '{"type":"tool_request","tool":"read_file","args":{"path":"README.md"},'
            '"bogus_field":true}',
            '{"type":"final_answer","message":"Nothing to run.","summary":"ok","citations":[]}',
        ],
    )
    result = await loop.run(
        agent=coding_agent(),
        message="inspect",
        context=context,
        request_id="request",
    )
    assert result.status == "ok"
    assert result.final_message == "Nothing to run."
    assert len(runtime.calls) == 2  # original + repair
    rows = await database.fetchall("SELECT * FROM tool_calls")
    assert rows == []  # the invalid tool_request was never executed
    await database.close()


@pytest.mark.asyncio
async def test_structured_agent_loop_rejects_blocked_tool(settings_tmp) -> None:
    loop, context, _memory, database, _runtime = await make_loop(
        settings_tmp,
        ['{"type":"tool_request","tool":"read_file","args":{"path":"README.md"}}'],
    )
    agent = coding_agent()
    blocked_config = agent.config.model_copy(
        update={"blocked_tools": {"read_file"}, "allowed_tools": {"read_file"}}
    )
    blocked_agent = type(agent)(blocked_config)
    result = await loop.run(
        agent=blocked_agent,
        message="read",
        context=context,
        request_id="request",
    )
    assert result.status == "error"
    assert "blocked tool" in result.final_message
    await database.close()


@pytest.mark.asyncio
async def test_structured_agent_loop_compacts_large_tool_evidence(settings_tmp) -> None:
    loop, _context, _memory, database, _runtime = await make_loop(settings_tmp, [])
    loop.max_tool_result_chars = 120
    result = ToolResult(
        ok=False,
        stderr="warning: retained diagnostic\n" + ("failure detail " * 80),
        data={"repository_state_digest": "state-a"},
        risk_level="read_only",
        permission_level=0,
    )

    rendered = loop._format_tool_result("test_runner", result)

    payload = json.loads(rendered)
    assert payload["data"] == {"truncated": True}
    assert payload["evidence"]["action"] == "test_runner"
    assert payload["evidence"]["repository_state_digest"] == "state-a"
    assert payload["evidence"]["truncated"] is True
    assert payload["output"].endswith("[TRUNCATED]")
    assert loop._format_tool_result("test_runner", None) == "test_runner: no result"
    await database.close()
