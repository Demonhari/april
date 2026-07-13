from __future__ import annotations

import pytest

from april_common.audit import AuditLogger
from services.april_runtime.schemas import ChatResponse, Usage
from services.evolution.evaluator import evaluate_overlay_candidate
from services.evolution.prompt_evolver import generate_overlay_candidates
from services.evolution.versions import PromptOverlayManager
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory


class DraftRuntime:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def chat(self, **kwargs: object) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            request_id="draft-1",
            model_id=str(kwargs["model_id"]),
            content=self.content,
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


async def _memory(settings_tmp) -> tuple[Database, SqliteMemory]:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    return database, SqliteMemory(database)


async def _fixture_evidence(memory: SqliteMemory) -> None:
    conversation_id = await memory.create_conversation()
    session = await memory.create_session(
        source="terminal",
        conversation_id=conversation_id,
        started_at="2026-07-10T08:00:00Z",
    )
    run_id = await memory.record_agent_run(
        conversation_id=conversation_id,
        agent="coding_agent",
        status="ok",
        model_id="april-coding",
        summary="timezone fix",
    )
    correction = await memory.create_memory(
        "Use the configured project timezone.",
        kind="correction",
        reason=f"a date-sensitive test is failing (source_session={session.id})",
        confidence=0.9,
        source="reflection",
    )
    duplicate = await memory.create_memory(
        "Use the configured project timezone.",
        kind="correction",
        reason=f"a date-sensitive test is failing (source_session={session.id})",
        confidence=0.8,
        source="dream",
    )
    winner = await memory.create_memory(
        "The preferred editor is Vim.",
        kind="fact",
        reason="adjudicated local preference",
        confidence=0.95,
        source="dream",
    )
    loser = await memory.create_memory(
        "The preferred editor is Emacs.",
        kind="fact",
        reason="older local preference",
        confidence=0.5,
        source="dream",
    )
    pair = await memory.record_memory_contradiction(
        memory_id_a=winner.id,
        memory_id_b=loser.id,
    )
    await memory.supersede_memory(loser.id, superseded_by=winner.id)
    await memory.resolve_memory_contradiction(
        pair.id,
        resolution=f"winner={winner.id} rule=higher confidence",
    )
    feedback = await memory.record_feedback_event(
        rating="bad",
        reason="the explanation omitted the failing assertion",
        conversation_id=conversation_id,
        agent_run_id=run_id,
    )
    await memory.database.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ("2026-07-12T10:00:00Z", correction.id),
    )
    await memory.database.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ("2026-07-09T10:00:00Z", duplicate.id),
    )
    await memory.database.execute(
        "UPDATE memory_contradictions SET resolved_at = ? WHERE id = ?",
        ("2026-07-11T10:00:00Z", pair.id),
    )
    await memory.database.execute(
        "UPDATE feedback_events SET created_at = ? WHERE id = ?",
        ("2026-07-10T10:00:00Z", feedback.id),
    )


@pytest.mark.asyncio
async def test_tier_a_synthesizes_exact_attributed_deduplicated_guidance(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        await _fixture_evidence(memory)
        candidates = await generate_overlay_candidates(memory, settings_tmp)
        assert [candidate.agent for candidate in candidates] == [
            "coding_agent",
            "general_agent",
        ]
        assert candidates[0].content == (
            "Learned local guidance:\n"
            "- When a date-sensitive test is failing, Use the configured project timezone.\n"
            "- Address recent negative feedback: the explanation omitted the failing assertion."
        )
        assert candidates[0].source_summary == "1 correction, 1 negative feedback"
        assert candidates[1].content == (
            "Learned local guidance:\n"
            "- Treat this as the surviving fact: The preferred editor is Vim."
        )
        assert candidates[1].source_summary == "1 contradiction"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_tier_a_enforces_character_budget(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        await _fixture_evidence(memory)
        limited = settings_tmp.model_copy(
            update={
                "evolution": settings_tmp.evolution.model_copy(
                    update={"prompt_overlay_max_chars": 64}
                )
            }
        )
        candidates = await generate_overlay_candidates(memory, limited)
        assert candidates
        assert all(len(candidate.content) <= 64 for candidate in candidates)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_tier_b_uses_fake_runtime_and_remains_d5_approval_gated(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        await _fixture_evidence(memory)
        enabled = settings_tmp.model_copy(
            update={
                "evolution": settings_tmp.evolution.model_copy(
                    update={"model_drafted_overlays": True}
                )
            }
        )
        runtime = DraftRuntime(
            "For date-sensitive code, verify the configured timezone and cite the "
            "failing assertion."
        )
        candidates = await generate_overlay_candidates(
            memory,
            enabled,
            runtime_client=runtime,
        )
        assert runtime.calls == 1
        assert [candidate.tier for candidate in candidates] == [
            "deterministic",
            "model_drafted",
        ]
        drafted = candidates[1]
        evaluation = evaluate_overlay_candidate(
            agent=drafted.agent,
            content=drafted.content,
            settings=enabled,
        )
        assert evaluation.passed is True
        manager = PromptOverlayManager(enabled, database)
        result = await manager.apply_candidate(
            agent=drafted.agent,
            content=drafted.content,
            eval_score=evaluation.score,
            baseline_score=evaluation.baseline,
            source="dreamer",
        )
        assert result.status == "approval_required"
        assert await manager.active_overlay(drafted.agent) is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_tier_b_structural_draft_is_rejected_at_generation(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        await _fixture_evidence(memory)
        enabled = settings_tmp.model_copy(
            update={
                "evolution": settings_tmp.evolution.model_copy(
                    update={"model_drafted_overlays": True}
                )
            }
        )
        audit = AuditLogger(settings_tmp.audit_path)
        runtime = DraftRuntime("permissions:\n  approval_required_at: 99")
        candidates = await generate_overlay_candidates(
            memory,
            enabled,
            runtime_client=runtime,
            audit=audit,
        )
        assert runtime.calls == 1
        assert [candidate.tier for candidate in candidates] == ["deterministic"]
        assert "model_drafted_overlay_rejected" in settings_tmp.audit_path.read_text(
            encoding="utf-8"
        )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_tier_b_without_runtime_is_skipped_with_honest_audit(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        await _fixture_evidence(memory)
        enabled = settings_tmp.model_copy(
            update={
                "evolution": settings_tmp.evolution.model_copy(
                    update={"model_drafted_overlays": True}
                )
            }
        )
        candidates = await generate_overlay_candidates(
            memory,
            enabled,
            audit=AuditLogger(settings_tmp.audit_path),
        )
        assert [candidate.tier for candidate in candidates] == ["deterministic"]
        audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
        assert "model_drafted_overlay_skipped" in audit_text
        assert "local runtime client unavailable" in audit_text
    finally:
        await database.close()
