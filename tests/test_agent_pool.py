from __future__ import annotations

import anyio
import pytest
from fastapi.testclient import TestClient

from services.api.server import create_app
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.pool.agent_pool import CALL_SIGNS, AgentPool
from tests.conftest import FakeRuntimeClient
from tests.test_core_api import auth, make_container


class RecordingRuntime(FakeRuntimeClient):
    def __init__(self, *, fail_load: bool = False) -> None:
        super().__init__()
        self.fail_load = fail_load
        self.loads: list[tuple[str, str | None]] = []

    async def load(self, model_id: str, *, request_id: str | None = None) -> dict[str, object]:
        self.loads.append((model_id, request_id))
        if self.fail_load:
            raise RuntimeError("load failed")
        return {
            "request_id": request_id or "test-request",
            "model_id": model_id,
            "state": "loaded",
            "message": "loaded",
        }


class FixedGovernor:
    def __init__(self, *, allowed: bool, reasons: tuple[str, ...] = ()) -> None:
        self.allowed = allowed
        self.reasons = reasons

    def assess_model_load(self, *, projected_resident_gb: float | None = None) -> object:
        del projected_resident_gb
        return type(
            "Decision",
            (),
            {"allowed": self.allowed, "reasons": self.reasons},
        )()


class RecordingAudit:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def write(self, payload: dict[str, object]) -> None:
        self.records.append(payload)


class RecordingPrewarmPool(AgentPool):
    def __init__(self, memory: SqliteMemory) -> None:
        super().__init__(memory)
        self.scheduled: list[tuple[str, str | None, str | None]] = []

    def schedule_prewarm(
        self,
        *,
        agent: str,
        model_id: str | None,
        request_id: str | None = None,
    ) -> None:
        self.scheduled.append((agent, model_id, request_id))
        return None


async def _memory(settings) -> tuple[Database, SqliteMemory]:
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    return database, SqliteMemory(database)


@pytest.mark.asyncio
async def test_scorecards_reflect_persisted_runs_and_feedback(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        conversation_id = await memory.create_conversation()
        ok_run = await memory.record_agent_run(
            conversation_id=conversation_id,
            agent="coding_agent",
            status="ok",
            model_id="april-coding",
            summary="fixed a bug",
        )
        await memory.record_agent_run(
            conversation_id=conversation_id,
            agent="coding_agent",
            status="error",
            model_id="april-coding",
            summary="failed run",
        )
        await memory.record_feedback_event(
            rating="good", reason=None, conversation_id=conversation_id, agent_run_id=ok_run
        )
        await memory.record_feedback_event(
            rating="bad",
            reason="too slow",
            conversation_id=conversation_id,
            agent_run_id=ok_run,
        )

        cards = {card.agent: card for card in await AgentPool(memory).scorecards()}
        forge = cards["coding_agent"]
        assert forge.call_sign == "Forge"
        assert forge.total_runs == 2
        assert forge.recent_runs == 2
        assert forge.ok_runs == 1
        assert forge.error_runs == 1
        assert forge.feedback_good == 1
        assert forge.feedback_bad == 1
        assert forge.last_run_at is not None

        # Agents with no persisted data report honest zeros, never made-up scores.
        sage = cards["reasoning_agent"]
        assert sage.call_sign == "Sage"
        assert sage.total_runs == 0
        assert sage.feedback_good == 0
        assert sage.last_run_at is None

        # Every known call sign appears exactly once.
        for agent, sign in CALL_SIGNS.items():
            assert cards[agent].call_sign == sign
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_scorecards_include_unknown_agents_from_data(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        conversation_id = await memory.create_conversation()
        await memory.record_agent_run(
            conversation_id=conversation_id,
            agent="playbook_runner",
            status="ok",
            model_id=None,
            summary="playbook",
        )
        cards = {card.agent: card for card in await AgentPool(memory).scorecards()}
        assert "playbook_runner" in cards
        # No configured call sign: the id doubles as the display name.
        assert cards["playbook_runner"].call_sign == "playbook_runner"
        assert cards["playbook_runner"].total_runs == 1
    finally:
        await database.close()


def test_pool_api_lists_registry_agents(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get("/pool/agents", headers=auth(settings_tmp))
    assert response.status_code == 200
    agents = response.json()["agents"]
    by_agent = {card["agent"]: card for card in agents}
    assert by_agent["coding_agent"]["call_sign"] == "Forge"
    assert by_agent["general_agent"]["call_sign"] == "Prime"
    assert all(isinstance(card["total_runs"], int) for card in agents)

    unauthorized = client.get("/pool/agents")
    assert unauthorized.status_code == 403


@pytest.mark.asyncio
async def test_prewarm_requests_selected_specialist_models(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        runtime = RecordingRuntime()
        audit = RecordingAudit()
        pool = AgentPool(
            memory,
            runtime_client=runtime,
            governor=FixedGovernor(allowed=True),
            audit=audit,
        )
        for agent, model_id in (
            ("coding_agent", "april-coding"),
            ("reading_agent", "april-reading"),
            ("reasoning_agent", "april-brain"),
        ):
            result = await pool.prewarm_selected(
                agent=agent,
                model_id=model_id,
                request_id=f"req-{agent}",
            )
            assert result.status == "loaded"
        assert [model for model, _request_id in runtime.loads] == [
            "april-coding",
            "april-reading",
            "april-brain",
        ]
        assert [record["status"] for record in audit.records if record["status"] == "loaded"] == [
            "loaded",
            "loaded",
            "loaded",
        ]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_prewarm_skips_under_governor_denial(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        runtime = RecordingRuntime()
        audit = RecordingAudit()
        pool = AgentPool(
            memory,
            runtime_client=runtime,
            governor=FixedGovernor(
                allowed=False,
                reasons=("projected_ram_headroom_below_policy",),
            ),
            audit=audit,
        )
        result = await pool.prewarm_selected(
            agent="coding_agent",
            model_id="april-coding",
            request_id="req-denied",
        )
        assert result.status == "skipped"
        assert runtime.loads == []
        assert audit.records[-1]["reason"] == "projected_ram_headroom_below_policy"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_prewarm_failure_is_audited_without_raising(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        audit = RecordingAudit()
        pool = AgentPool(
            memory,
            runtime_client=RecordingRuntime(fail_load=True),
            governor=FixedGovernor(allowed=True),
            audit=audit,
        )
        result = await pool.prewarm_selected(
            agent="reading_agent",
            model_id="april-reading",
            request_id="req-fail",
        )
        assert result.status == "failed"
        assert audit.records[-1]["status"] == "failed"
        assert audit.records[-1]["reason"] == "RuntimeError"
    finally:
        await database.close()


def test_brain_routes_schedule_specialist_prewarm(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    pool = RecordingPrewarmPool(container.memory)
    container.orchestrator.agent_pool = pool

    async def run_turns() -> None:
        await container.orchestrator.chat("inspect the animation code", request_id="req-code")
        await container.orchestrator.chat("summarize README", request_id="req-read")
        await container.orchestrator.chat(
            "reason through the trade-off",
            request_id="req-reason",
        )

    anyio.run(run_turns)

    scheduled = [(agent, model_id) for agent, model_id, _request_id in pool.scheduled]
    assert ("coding_agent", "april-coding") in scheduled
    assert ("reading_agent", "april-reading") in scheduled
    assert any(agent == "reasoning_agent" for agent, _model_id in scheduled)
