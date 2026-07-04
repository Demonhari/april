from __future__ import annotations

import anyio
import pytest
from fastapi.testclient import TestClient

from services.api.server import create_app
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.pool.agent_pool import CALL_SIGNS, AgentPool
from tests.test_core_api import auth, make_container


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
