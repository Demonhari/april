from __future__ import annotations

import pytest

from agents.registry import default_agent_registry
from april_common.audit import AuditLogger
from april_common.errors import PermissionDeniedError
from services.brain.orchestrator import AprilOrchestrator
from services.brain.schemas import BrainDecision
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from skills.registry import default_registry
from tests.conftest import FakeRuntimeClient


class UnknownAgentRouter:
    async def route(self, message: str, *, request_id: str | None = None) -> BrainDecision:
        return BrainDecision(
            intent="bad",
            agent="missing_agent",
            model_id="april-brain",
            tools_needed=[],
            memory_queries=[],
            permission_level=0,
            risk_level="none",
            needs_confirmation=False,
            task_steps=[],
            decision_summary="bad",
        )


@pytest.mark.asyncio
async def test_unknown_agent_rejected(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    registry = default_registry()
    orchestrator = AprilOrchestrator(
        settings=settings_tmp,
        runtime_client=FakeRuntimeClient(),
        memory=SqliteMemory(database),
        tool_registry=registry,
        permission_engine=PermissionEngine(registry),
        approvals=ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60),
        agent_registry=default_agent_registry(),
        brain_router=UnknownAgentRouter(),  # type: ignore[arg-type]
    )
    with pytest.raises(PermissionDeniedError):
        await orchestrator.chat("hello")
    await database.close()
