from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from april_common.audit import AuditLogger
from services.api.server import create_app
from services.evolution.dreamer import DreamerService
from services.evolution.scheduler import EvolutionSchedulerGate
from services.evolution.versions import PromptOverlayManager
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.pool.governor import ResourceGovernor, ResourcePolicy, ResourceSignals
from tests.test_core_api import auth, make_container


class FixedSignals:
    def __init__(self, signals: ResourceSignals) -> None:
        self.signals = signals

    def sample(self) -> ResourceSignals:
        return self.signals


async def _manager(settings_tmp) -> tuple[Database, PromptOverlayManager]:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    audit = AuditLogger(settings_tmp.audit_path)
    return database, PromptOverlayManager(settings_tmp, database, audit=audit)


@pytest.mark.asyncio
async def test_evolution_write_guard_fences_paths_and_audits(settings_tmp) -> None:
    audit = AuditLogger(settings_tmp.audit_path)
    guard = EvolutionWriteGuard(settings_tmp, audit=audit)
    allowed = guard.write_text(settings_tmp.evolution_path / "ok.txt", "ok")
    assert allowed.exists()
    with pytest.raises(PermissionError):
        guard.write_text(settings_tmp.home / "README.md", "bad")
    assert "evolution_write_guard_violation" in settings_tmp.audit_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_prompt_overlay_rejects_malicious_structural_change(settings_tmp) -> None:
    database, manager = await _manager(settings_tmp)
    try:
        result = await manager.apply_candidate(
            agent="general_agent",
            content="permissions:\n  approval_required_at: 99",
            eval_score=1.0,
            baseline_score=0.5,
        )
        assert result.status == "discarded"
        assert await manager.versions(agent="general_agent") == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_prompt_overlay_discards_below_baseline_and_requires_hand_approval(
    settings_tmp,
) -> None:
    database, manager = await _manager(settings_tmp)
    try:
        low = await manager.apply_candidate(
            agent="general_agent",
            content="Prefer shorter answers.",
            eval_score=0.4,
            baseline_score=0.5,
        )
        assert low.status == "discarded"
        low_hand = await manager.apply_candidate(
            agent="general_agent",
            content="Prefer careful answers.",
            eval_score=0.4,
            baseline_score=0.5,
            source="hand",
        )
        assert low_hand.status == "discarded"
        hand = await manager.apply_candidate(
            agent="general_agent",
            content="Prefer precise answers.",
            eval_score=0.9,
            baseline_score=0.5,
            source="hand",
        )
        assert hand.status == "approval_required"
        assert await manager.versions(agent="general_agent") == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_prompt_overlay_rollback_restores_byte_identical_version(settings_tmp) -> None:
    database, manager = await _manager(settings_tmp)
    try:
        first = await manager.apply_candidate(
            agent="general_agent",
            content="First overlay.",
            eval_score=0.8,
            baseline_score=0.5,
        )
        second = await manager.apply_candidate(
            agent="general_agent",
            content="Second overlay.",
            eval_score=0.9,
            baseline_score=0.5,
        )
        assert first.status == "applied"
        assert second.status == "applied"
        assert await manager.active_overlay("general_agent") == b"Second overlay."
        rollback = await manager.rollback(agent="general_agent", version=1)
        assert rollback.status == "applied"
        assert await manager.active_overlay("general_agent") == b"First overlay."
        assert first.path is not None
        assert first.path.read_bytes() == b"First overlay."
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_deleting_evolution_data_returns_stock_behavior(settings_tmp) -> None:
    database, manager = await _manager(settings_tmp)
    try:
        await manager.apply_candidate(
            agent="general_agent",
            content="Temporary overlay.",
            eval_score=0.8,
            baseline_score=0.5,
        )
        shutil.rmtree(settings_tmp.evolution_path)
        assert await manager.active_overlay("general_agent") is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dreamer_scheduler_gate_and_report(settings_tmp) -> None:
    enabled = settings_tmp.model_copy(
        update={
            "evolution": settings_tmp.evolution.model_copy(
                update={"enabled": True, "window": "02:00-04:00"}
            )
        }
    )
    database = Database(enabled.database_path)
    await database.connect()
    await run_migrations(database)
    try:
        memory = SqliteMemory(database)
        governor = ResourceGovernor(
            enabled,
            provider=FixedSignals(
                ResourceSignals(
                    ram_headroom_gb=12.0,
                    cpu_load_percent=5.0,
                    on_ac_power=True,
                    user_idle_seconds=600.0,
                )
            ),
            policy=ResourcePolicy(min_ram_headroom_gb=1.0, max_cpu_load_percent=90.0),
        )
        gate = EvolutionSchedulerGate(enabled, memory, governor=governor)
        service = DreamerService(enabled, memory=memory, gate=gate)
        now = datetime(2026, 7, 3, 2, 30, tzinfo=UTC)
        result = await service.run_once(now)
        assert result.status == "completed"
        assert result.report_path is not None
        assert "no evolution candidates" in Path(result.report_path).read_text(
            encoding="utf-8"
        )
        second = await service.run_once(now)
        assert second.status == "skipped"
        assert second.reason == "already ran today"
    finally:
        await database.close()


def test_evolution_api_rollback(settings_tmp) -> None:
    import anyio

    async def apply_overlay(content: str, score: float):
        return await manager.apply_candidate(
            agent="general_agent",
            content=content,
            eval_score=score,
            baseline_score=0.5,
        )

    container = anyio.run(make_container, settings_tmp)
    manager = PromptOverlayManager(settings_tmp, container.database)
    first = anyio.run(apply_overlay, "First API overlay.", 0.8)
    anyio.run(apply_overlay, "Second API overlay.", 0.9)
    client = TestClient(create_app(container))
    response = client.post(
        "/evolution/rollback",
        json={"agent": "general_agent", "version": first.version},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["rollback"]["status"] == "applied"
    assert anyio.run(manager.active_overlay, "general_agent") == b"First API overlay."
