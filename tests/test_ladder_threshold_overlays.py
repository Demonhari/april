from __future__ import annotations

import json
import shutil

import pytest

from april_common.settings import DeepModeSettings
from services.evolution.versions import (
    LadderThresholdOverlayManager,
    LadderThresholds,
    active_ladder_thresholds,
    bounded_ladder_threshold_nudge,
    propose_ladder_thresholds_from_memory,
)
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory


def test_deep_mode_threshold_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="deep_mode thresholds"):
        DeepModeSettings(deep_confidence_threshold=0.8, verified_confidence_threshold=0.7)
    with pytest.raises(ValueError, match="deep_mode thresholds"):
        DeepModeSettings(deep_confidence_threshold=0.0, verified_confidence_threshold=0.7)
    valid = DeepModeSettings(deep_confidence_threshold=0.45, verified_confidence_threshold=0.75)
    assert valid.deep_confidence_threshold == 0.45


def test_ladder_threshold_nudge_bounds_and_clamps() -> None:
    high = bounded_ladder_threshold_nudge(
        LadderThresholds(0.58, 0.88),
        nudge=0.5,
    )
    assert high == LadderThresholds(0.6, 0.9)
    low = bounded_ladder_threshold_nudge(
        LadderThresholds(0.22, 0.52),
        nudge=-0.5,
    )
    assert low == LadderThresholds(0.2, 0.5)


def test_ladder_overlay_rejects_malicious_keys_with_zero_effect(settings_tmp) -> None:
    guard = EvolutionWriteGuard(settings_tmp)
    guard.write_text(
        settings_tmp.evolution_path / "config" / "ladder-v001.json",
        json.dumps(
            {
                "deep_confidence_threshold": 0.5,
                "verified_confidence_threshold": 0.8,
                "permissions": {"approval_required_at": 99},
            }
        ),
    )
    guard.write_text(
        settings_tmp.evolution_path / "config" / "ladder-active.json",
        json.dumps({"schema_version": 1, "active_version": 1}),
    )
    assert active_ladder_thresholds(settings_tmp) == {
        "deep_confidence_threshold": 0.4,
        "verified_confidence_threshold": 0.7,
    }


def test_ladder_overlay_below_baseline_is_discarded(settings_tmp) -> None:
    manager = LadderThresholdOverlayManager(settings_tmp)
    result = manager.apply_candidate(
        LadderThresholds(0.45, 0.75),
        eval_score=0.4,
        baseline_score=0.5,
    )
    assert result.status == "discarded"
    assert not (settings_tmp.evolution_path / "config" / "ladder-v001.json").exists()
    assert active_ladder_thresholds(settings_tmp)["deep_confidence_threshold"] == 0.4


def test_ladder_overlay_rollback_and_delete_restore_thresholds(settings_tmp) -> None:
    manager = LadderThresholdOverlayManager(settings_tmp)
    first = manager.apply_candidate(
        LadderThresholds(0.45, 0.75),
        eval_score=1.0,
        baseline_score=1.0,
    )
    second = manager.apply_candidate(
        LadderThresholds(0.5, 0.8),
        eval_score=1.0,
        baseline_score=1.0,
    )
    assert first.status == "applied"
    assert second.status == "applied"
    assert active_ladder_thresholds(settings_tmp) == {
        "deep_confidence_threshold": 0.5,
        "verified_confidence_threshold": 0.8,
    }
    rollback = manager.rollback()
    assert rollback.status == "applied"
    assert active_ladder_thresholds(settings_tmp) == {
        "deep_confidence_threshold": 0.45,
        "verified_confidence_threshold": 0.75,
    }
    shutil.rmtree(settings_tmp.evolution_path)
    assert active_ladder_thresholds(settings_tmp) == {
        "deep_confidence_threshold": 0.4,
        "verified_confidence_threshold": 0.7,
    }


@pytest.mark.asyncio
async def test_ladder_threshold_proposal_uses_rung_outcomes_and_feedback(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    try:
        conversation_id = await memory.create_conversation()
        run_id = await memory.record_agent_run(
            conversation_id=conversation_id,
            agent="general_agent",
            status="ok",
            model_id="april-brain",
            summary="standard answer",
            metadata={"intelligence_rung": 1},
        )
        await memory.record_feedback_event(
            rating="bad",
            conversation_id=conversation_id,
            agent_run_id=run_id,
        )
        proposed = await propose_ladder_thresholds_from_memory(settings_tmp, memory)
        assert proposed == LadderThresholds(0.45, 0.75)
    finally:
        await database.close()
