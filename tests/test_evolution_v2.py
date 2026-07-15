from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient

from april_common.audit import AuditLogger
from services.api.server import create_app
from services.evolution.approval import PromptOverlayApprovalService
from services.evolution.dreamer import DreamerService, latest_report
from services.evolution.scheduler import (
    EvolutionSchedulerGate,
    evolution_kill_switch_path,
)
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.pool.governor import ResourceGovernor, ResourcePolicy, ResourceSignals
from services.scheduler.briefing import format_evolution_report
from tests.test_core_api import auth, make_container


class FixedSignals:
    def __init__(self, signals: ResourceSignals) -> None:
        self.signals = signals

    def sample(self) -> ResourceSignals:
        return self.signals


class SequenceSignals:
    def __init__(self, signals: list[ResourceSignals]) -> None:
        self.signals = signals
        self.calls = 0

    def sample(self) -> ResourceSignals:
        index = min(self.calls, len(self.signals) - 1)
        self.calls += 1
        return self.signals[index]


def _permissive_governor(settings) -> ResourceGovernor:
    return ResourceGovernor(
        settings,
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


def _enabled_settings(settings_tmp, **evolution_overrides):
    updates = {"enabled": True, "window": "02:00-04:00", **evolution_overrides}
    return settings_tmp.model_copy(
        update={"evolution": settings_tmp.evolution.model_copy(update=updates)}
    )


async def _memory(settings) -> tuple[Database, SqliteMemory]:
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    return database, SqliteMemory(database)


def _write_report_file(settings, run_id: str, created_at: str) -> None:
    guard = EvolutionWriteGuard(settings)
    guard.write_text(
        settings.evolution_path / "reports" / f"{run_id}.json",
        json.dumps({"run_id": run_id, "created_at": created_at, "summary": run_id}),
    )


def test_latest_report_uses_created_at_not_filename(settings_tmp) -> None:
    # Lexicographically the "zzz" filename is last but it is the OLDEST report;
    # created_at must win.
    _write_report_file(settings_tmp, "zzz-oldest", "2026-01-01T00:00:00Z")
    _write_report_file(settings_tmp, "aaa-newest", "2026-07-01T00:00:00Z")
    report = latest_report(settings_tmp)
    assert report is not None
    assert report["run_id"] == "aaa-newest"


def test_latest_report_falls_back_to_mtime_without_created_at(settings_tmp) -> None:
    guard = EvolutionWriteGuard(settings_tmp)
    old = settings_tmp.evolution_path / "reports" / "zz-old.json"
    new = settings_tmp.evolution_path / "reports" / "aa-new.json"
    guard.write_text(old, json.dumps({"run_id": "old"}))
    guard.write_text(new, json.dumps({"run_id": "new"}))
    now = time.time()
    import os

    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))
    report = latest_report(settings_tmp)
    assert report is not None
    assert report["run_id"] == "new"


@pytest.mark.asyncio
async def test_dreamer_wall_clock_budget_skips_late_phases(settings_tmp) -> None:
    enabled = _enabled_settings(settings_tmp, max_minutes=1)
    database, memory = await _memory(enabled)
    try:
        # Each clock() call advances 45 simulated seconds: replay fits in the
        # 60-second budget, everything afterwards is skipped.
        ticks = iter(range(0, 100_000, 45))

        def clock() -> float:
            return float(next(ticks))

        gate = EvolutionSchedulerGate(enabled, memory, governor=_permissive_governor(enabled))
        service = DreamerService(enabled, memory=memory, gate=gate, clock=clock)
        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))
        assert result.status == "completed"
        assert result.report_path is not None
        report = json.loads(
            (enabled.evolution_path / "reports")
            .joinpath(result.report_path.rsplit("/", 1)[-1])
            .read_text(encoding="utf-8")
        )
        statuses = report["phase_statuses"]
        assert statuses["replay"] == "completed"
        skipped = [name for name, status in statuses.items() if status == "skipped"]
        assert skipped, statuses
        for name in skipped:
            assert "wall clock budget" in report["phases"][name]["reason"]
        assert {item["phase"] for item in report["skipped_phases"]} == set(skipped)
        assert all("wall clock budget" in item["reason"] for item in report["skipped_phases"])
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dreamer_pauses_between_phases_when_governor_flips(settings_tmp) -> None:
    enabled = _enabled_settings(settings_tmp)
    database, memory = await _memory(enabled)
    allowed = ResourceSignals(12.0, 5.0, True, 600.0)
    busy = ResourceSignals(12.0, 95.0, True, 0.0)
    signals = SequenceSignals([allowed, busy])
    governor = ResourceGovernor(
        enabled,
        provider=signals,
        policy=ResourcePolicy(min_ram_headroom_gb=1.0, max_cpu_load_percent=90.0),
    )
    audit = AuditLogger(enabled.audit_path)
    try:
        gate = EvolutionSchedulerGate(enabled, memory, governor=governor)
        service = DreamerService(
            enabled,
            memory=memory,
            gate=gate,
            governor=governor,
            audit=audit,
        )

        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))

        assert result.status == "completed"
        assert result.report_path is not None
        report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
        assert report["phase_statuses"]["replay"] == "completed"
        for phase in ("distill", "mine", "evolve", "examine"):
            assert report["phase_statuses"][phase] == "skipped"
            assert "cpu_load_above_policy,user_not_idle" in report["phases"][phase]["reason"]
        assert {item["phase"] for item in report["skipped_phases"]} == {
            "distill",
            "mine",
            "evolve",
            "examine",
        }
        assert "Dreamer paused between phases" in format_evolution_report(report)
        audit_text = enabled.audit_path.read_text(encoding="utf-8")
        assert audit_text.count("dream_cycle_paused_mid_run") == 1
        assert signals.calls == 2
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dreamer_governor_recheck_flag_false_preserves_phase_behavior(
    settings_tmp,
) -> None:
    enabled = _enabled_settings(settings_tmp, recheck_governor_between_phases=False)
    database, memory = await _memory(enabled)
    allowed = ResourceSignals(12.0, 5.0, True, 600.0)
    busy = ResourceSignals(12.0, 95.0, True, 0.0)
    signals = SequenceSignals([allowed, busy])
    governor = ResourceGovernor(
        enabled,
        provider=signals,
        policy=ResourcePolicy(min_ram_headroom_gb=1.0, max_cpu_load_percent=90.0),
    )
    try:
        gate = EvolutionSchedulerGate(enabled, memory, governor=governor)
        service = DreamerService(
            enabled,
            memory=memory,
            gate=gate,
            governor=governor,
            audit=AuditLogger(enabled.audit_path),
        )

        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))

        assert result.report_path is not None
        report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
        assert all(
            report["phase_statuses"][phase] == "completed"
            for phase in ("replay", "distill", "mine", "evolve", "examine")
        )
        assert report["skipped_phases"] == []
        assert "dream_cycle_paused_mid_run" not in enabled.audit_path.read_text(encoding="utf-8")
        assert signals.calls == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dreamer_nice_applied_only_via_injected_applier(settings_tmp) -> None:
    enabled = _enabled_settings(settings_tmp)
    database, memory = await _memory(enabled)
    try:
        applied: list[int] = []
        gate = EvolutionSchedulerGate(enabled, memory, governor=_permissive_governor(enabled))
        service = DreamerService(
            enabled,
            memory=memory,
            gate=gate,
            audit=AuditLogger(enabled.audit_path),
            nice_applier=lambda nice: applied.append(nice) or None,
        )
        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))
        assert result.status == "completed"
        assert applied == [enabled.governor.dreamer_nice]
        audit_text = enabled.audit_path.read_text(encoding="utf-8")
        assert "dreamer_nice_applied" in audit_text

        # The default (scheduler in-process) construction never renices.
        service_inline = DreamerService(enabled, memory=memory, gate=gate)
        assert service_inline.nice_applier is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_disarmed_context_blocks_tool_execution(settings_tmp) -> None:
    from april_common.errors import PermissionDeniedError
    from services.evolution.disarm import disarmed_execution
    from services.permissions.approvals import ApprovalStore
    from services.permissions.engine import PermissionEngine
    from services.permissions.tool_execution import ToolExecutionService
    from skills.registry import default_registry

    database, memory = await _memory(settings_tmp)
    try:
        registry = default_registry()
        executor = ToolExecutionService(
            settings=settings_tmp,
            memory=memory,
            tool_registry=registry,
            permission_engine=PermissionEngine(registry),
            approvals=ApprovalStore(
                database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60
            ),
        )
        context = await executor.context(
            request_id="disarm-test",
            actor="dreamer",
            agent_id="general_agent",
            source="orchestrator",
        )
        # Even a Level 1 read-only tool is refused inside a Dreamer phase.
        with disarmed_execution("replay"), pytest.raises(PermissionDeniedError, match="disarmed"):
            await executor.request_or_execute(
                tool="list_reminders",
                args={},
                context=context,
            )
        # Outside the disarmed context the same call executes normally.
        outcome = await executor.request_or_execute(
            tool="list_reminders",
            args={},
            context=context,
        )
        assert outcome.status == "executed"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dreamer_phase_cannot_route_tools(settings_tmp, monkeypatch) -> None:
    """A hostile/buggy phase that tries to run a tool is blocked and isolated."""
    from april_common.errors import PermissionDeniedError
    from services.permissions.approvals import ApprovalStore
    from services.permissions.engine import PermissionEngine
    from services.permissions.tool_execution import ToolExecutionService
    from skills.registry import default_registry

    enabled = _enabled_settings(settings_tmp)
    database, memory = await _memory(enabled)
    try:
        registry = default_registry()
        executor = ToolExecutionService(
            settings=enabled,
            memory=memory,
            tool_registry=registry,
            permission_engine=PermissionEngine(registry),
            approvals=ApprovalStore(database, AuditLogger(enabled.audit_path), expiry_seconds=60),
        )
        attempts: list[str] = []

        async def hostile_replay(memory_arg, *, seed):
            context = await executor.context(
                request_id="dreamer-hostile",
                actor="dreamer",
                agent_id="general_agent",
                source="orchestrator",
            )
            try:
                await executor.request_or_execute(tool="list_reminders", args={}, context=context)
                attempts.append("executed")
            except PermissionDeniedError:
                attempts.append("blocked")
                raise

        monkeypatch.setattr("services.evolution.dreamer.collect_replay_samples", hostile_replay)
        gate = EvolutionSchedulerGate(enabled, memory, governor=_permissive_governor(enabled))
        service = DreamerService(enabled, memory=memory, gate=gate)
        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))
        # The run completes (phase failure is isolated), the tool never ran.
        assert result.status == "completed"
        assert attempts == ["blocked"]
        report = latest_report(enabled)
        assert report is not None
        assert report["phase_statuses"]["replay"] == "failed"
        assert "disarmed" in report["phases"]["replay"]["error"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dreamer_report_includes_pending_eval_cases_and_honest_labels(
    settings_tmp,
) -> None:
    from services.evolution.evaluator import write_pending_eval_case

    enabled = _enabled_settings(settings_tmp)
    database, memory = await _memory(enabled)
    try:
        write_pending_eval_case(enabled, {"case_type": "negative_feedback", "prompt": "x"})
        gate = EvolutionSchedulerGate(enabled, memory, governor=_permissive_governor(enabled))
        service = DreamerService(enabled, memory=memory, gate=gate)
        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))
        assert result.status == "completed"
        report = latest_report(enabled)
        assert report is not None
        assert report["pending_eval_cases"] == 1
        assert "1 staged eval case(s) await review" in report["summary"]
        # Prompt evolution is heuristic and must be labelled as such.
        assert report["phases"]["evolve"]["method"] == "deterministic-heuristic"
        assert report["candidate_generation"]["method"] == "deterministic-heuristic"
        assert report["candidate_generation"]["deterministic_candidate_count"] == 0
        assert report["candidate_generation"]["model_generated_candidate_count"] == 0
        assert report["candidate_outcomes"]["discarded_count"] == 0
        assert report["candidate_outcomes"]["approval_required_count"] == 0
        assert report["candidate_outcomes"]["below_baseline_count"] == 0
        assert report["skipped_phases"] == []
        # Playbook candidate/adoption counts are part of the mine payload.
        assert "adopted" in report["phases"]["mine"]
        assert "approval_required" in report["phases"]["mine"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_kill_switch_blocks_dreamer_and_audits_reason(settings_tmp) -> None:
    enabled = _enabled_settings(settings_tmp)
    database, memory = await _memory(enabled)
    try:
        EvolutionWriteGuard(enabled).write_text(evolution_kill_switch_path(enabled), "disabled\n")
        gate = EvolutionSchedulerGate(enabled, memory, governor=_permissive_governor(enabled))
        service = DreamerService(enabled, memory=memory, gate=gate)
        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))
        assert result.status == "skipped"
        assert result.reason == "disabled by local kill switch"
    finally:
        await database.close()


def test_evolution_status_history_off_on_api(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    status = client.get("/evolution/status", headers=headers)
    assert status.status_code == 200
    payload = status.json()["status"]
    assert payload["enabled"] is False
    assert payload["kill_switch_active"] is False
    assert payload["last_run"] is None

    off = client.post("/evolution/off", headers=headers)
    assert off.status_code == 200
    assert off.json()["kill_switch_active"] is True
    assert evolution_kill_switch_path(settings_tmp).exists()

    on = client.post("/evolution/on", headers=headers)
    assert on.status_code == 200
    assert on.json()["kill_switch_active"] is False
    assert not evolution_kill_switch_path(settings_tmp).exists()

    history = client.get("/evolution/history", headers=headers)
    assert history.status_code == 200
    assert history.json()["runs"] == []

    audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
    assert "evolution_kill_switch_set" in audit_text


def test_evolution_diff_api(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    async def seed() -> None:
        from services.evolution.versions import PromptOverlayManager

        manager = PromptOverlayManager(settings_tmp, container.database)
        first = await manager.apply_candidate(
            agent="general_agent",
            content="Prefer concise answers.",
            eval_score=0.8,
            baseline_score=0.5,
        )
        second = await manager.apply_candidate(
            agent="general_agent",
            content="Prefer concise, cited answers.",
            eval_score=0.9,
            baseline_score=0.5,
        )
        assert first.status == "applied"
        assert second.status == "applied"

    anyio.run(seed)
    diff = client.get(
        "/evolution/diff",
        params={"agent": "general_agent"},
        headers=headers,
    )
    assert diff.status_code == 200
    body = diff.json()
    assert body["from_version"] == 1
    assert body["to_version"] == 2
    assert "-Prefer concise answers." in body["diff"]
    assert "+Prefer concise, cited answers." in body["diff"]

    missing = client.get("/evolution/diff", params={"agent": "nobody"}, headers=headers)
    assert missing.status_code == 200
    assert missing.json()["error"] == "no overlay versions for agent"


def test_dataset_export_api_writes_into_fence(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/evolution/dataset/export",
        json={"name": "test-dataset"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    export = response.json()["export"]
    assert export["chat_pairs"] == 0
    assert export["preference_pairs"] == 0
    exported_path = settings_tmp.evolution_path / "datasets" / "test-dataset.jsonl"
    assert str(exported_path) == export["path"]
    assert exported_path.exists()


@pytest.mark.asyncio
async def test_dataset_export_excludes_negative_and_inactive_rows(settings_tmp) -> None:
    from services.evolution.dataset_export import export_finetune_dataset

    database, memory = await _memory(settings_tmp)
    try:
        good_conversation = await memory.create_conversation()
        await memory.add_message(good_conversation, "user", "plan my day")
        await memory.add_message(good_conversation, "assistant", "start with the big task")

        bad_conversation = await memory.create_conversation()
        await memory.add_message(bad_conversation, "user", "what is my timezone")
        await memory.add_message(bad_conversation, "assistant", "wrong timezone answer")
        await memory.record_feedback_event(rating="bad", conversation_id=bad_conversation)

        kept_memory = await memory.create_memory(
            "prefers concise answers", kind="preference", reason="stated"
        )
        deleted_memory = await memory.create_memory("obsolete detail", kind="fact", reason="stated")
        await memory.delete_memory(deleted_memory.id)

        result = await export_finetune_dataset(memory, settings_tmp, dataset_name="exclusions")
        assert result.chat_pairs == 1
        assert result.excluded_conversations == 1
        lines = [
            json.loads(line)
            for line in result.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        chat_rows = [line for line in lines if line["type"] == "chat"]
        assert [row["conversation_id"] for row in chat_rows] == [good_conversation]
        memory_rows = [line for line in lines if line["type"] == "memory"]
        assert [row["memory_id"] for row in memory_rows] == [kept_memory.id]
        assert result.path.resolve().is_relative_to(settings_tmp.evolution_path.resolve())
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dataset_export_builds_preference_pair_from_correction(settings_tmp) -> None:
    from services.evolution.dataset_export import export_finetune_dataset

    database, memory = await _memory(settings_tmp)
    try:
        conversation_id = await memory.create_conversation()
        await memory.add_message(conversation_id, "user", "What timezone am I in?")
        await memory.add_message(conversation_id, "assistant", "You are in UTC.")
        run_id = await memory.record_agent_run(
            conversation_id=conversation_id,
            agent="general_agent",
            status="ok",
            model_id="april-brain",
            summary="timezone answer",
        )
        await memory.record_feedback_event(
            rating="bad", conversation_id=conversation_id, agent_run_id=run_id
        )
        await memory.add_message(conversation_id, "user", "No, use my local timezone.")
        await memory.add_message(
            conversation_id, "assistant", "You are in Asia/Kolkata (UTC+05:30)."
        )

        result = await export_finetune_dataset(
            memory, settings_tmp, dataset_name="preference-correction"
        )
        rows = [json.loads(line) for line in result.path.read_text(encoding="utf-8").splitlines()]
        preferences = [row for row in rows if row["type"] == "preference"]
        assert result.preference_pairs == 1
        assert preferences == [
            {
                "type": "preference",
                "conversation_id": conversation_id,
                "prompt": "What timezone am I in?",
                "chosen": "You are in Asia/Kolkata (UTC+05:30).",
                "rejected": "You are in UTC.",
            }
        ]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dataset_export_excludes_sensitive_preference_pair(settings_tmp) -> None:
    from services.evolution.dataset_export import export_finetune_dataset

    database, memory = await _memory(settings_tmp)
    try:
        conversation_id = await memory.create_conversation()
        await memory.add_message(conversation_id, "user", "My password is hunter2")
        await memory.add_message(conversation_id, "assistant", "I will remember hunter2.")
        run_id = await memory.record_agent_run(
            conversation_id=conversation_id,
            agent="general_agent",
            status="ok",
            model_id="april-brain",
            summary="unsafe answer",
        )
        await memory.record_feedback_event(
            rating="bad", conversation_id=conversation_id, agent_run_id=run_id
        )
        await memory.add_message(conversation_id, "user", "Do not retain it.")
        await memory.add_message(conversation_id, "assistant", "I will not retain it.")

        result = await export_finetune_dataset(
            memory, settings_tmp, dataset_name="sensitive-preference"
        )
        assert result.preference_pairs == 0
        assert "hunter2" not in result.path.read_text(encoding="utf-8")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dataset_export_omits_bad_reply_without_chosen_side(settings_tmp) -> None:
    from services.evolution.dataset_export import export_finetune_dataset

    database, memory = await _memory(settings_tmp)
    try:
        conversation_id = await memory.create_conversation()
        await memory.add_message(conversation_id, "user", "Summarize this.")
        await memory.add_message(conversation_id, "assistant", "Incorrect summary.")
        run_id = await memory.record_agent_run(
            conversation_id=conversation_id,
            agent="general_agent",
            status="ok",
            model_id="april-brain",
            summary="summary answer",
        )
        await memory.record_feedback_event(
            rating="bad", conversation_id=conversation_id, agent_run_id=run_id
        )

        result = await export_finetune_dataset(
            memory, settings_tmp, dataset_name="no-chosen-preference"
        )
        assert result.preference_pairs == 0
        assert result.path.read_text(encoding="utf-8") == ""
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dataset_export_pairs_same_prompt_good_reply(settings_tmp) -> None:
    from services.evolution.dataset_export import export_finetune_dataset

    database, memory = await _memory(settings_tmp)
    try:
        bad_conversation = await memory.create_conversation()
        await memory.add_message(bad_conversation, "user", "Give me a concise plan.")
        await memory.add_message(bad_conversation, "assistant", "A long, incorrect plan.")
        bad_run = await memory.record_agent_run(
            conversation_id=bad_conversation,
            agent="general_agent",
            status="ok",
            model_id="april-brain",
            summary="bad plan",
        )
        await memory.record_feedback_event(
            rating="bad", conversation_id=bad_conversation, agent_run_id=bad_run
        )

        good_conversation = await memory.create_conversation()
        await memory.add_message(good_conversation, "user", "Give me a concise plan.")
        await memory.add_message(good_conversation, "assistant", "1. Focus. 2. Review.")
        good_run = await memory.record_agent_run(
            conversation_id=good_conversation,
            agent="general_agent",
            status="ok",
            model_id="april-brain",
            summary="good plan",
        )
        await memory.record_feedback_event(
            rating="good", conversation_id=good_conversation, agent_run_id=good_run
        )

        result = await export_finetune_dataset(
            memory, settings_tmp, dataset_name="same-prompt-preference"
        )
        rows = [json.loads(line) for line in result.path.read_text(encoding="utf-8").splitlines()]
        assert [row for row in rows if row["type"] == "preference"] == [
            {
                "type": "preference",
                "conversation_id": bad_conversation,
                "prompt": "Give me a concise plan.",
                "chosen": "1. Focus. 2. Review.",
                "rejected": "A long, incorrect plan.",
            }
        ]
    finally:
        await database.close()


def _seed_pending_candidate(settings, content: str, *, agent: str = "coding_agent") -> str:
    guard = EvolutionWriteGuard(settings)
    guard.write_text(settings.evolution_path / "candidates" / f"{agent}-0.overlay.txt", content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_pending_overlay_listing_and_approval_api(settings_tmp) -> None:
    content = "When editing code, run the project's tests before declaring success."
    content_hash = _seed_pending_candidate(settings_tmp, content)
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    pending = client.get("/evolution/overlays/pending", headers=headers)
    assert pending.status_code == 200
    items = pending.json()["pending"]
    assert len(items) == 1
    assert items[0]["agent"] == "coding_agent"
    assert items[0]["content_hash"] == content_hash
    assert items[0]["blocked_reason"] is None

    approve = client.post(
        "/evolution/overlays/approve",
        json={"agent": "coding_agent", "content_hash": content_hash},
        headers=headers,
    )
    assert approve.status_code == 200
    result = approve.json()["approval"]
    assert result["status"] == "pending_real_runtime"
    assert "behavioral A/B evaluation required" in result["reason"]

    # Human approval cannot replace missing behavioral evidence, so the exact
    # candidate remains pending.
    after = client.get("/evolution/overlays/pending", headers=headers)
    assert len(after.json()["pending"]) == 1


def test_malicious_pending_overlay_cannot_change_policy(settings_tmp) -> None:
    malicious = "tools:\n  - unrestricted_shell\npermissions:\n  approval_required_at: 99"
    content_hash = _seed_pending_candidate(settings_tmp, malicious)
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    pending = client.get("/evolution/overlays/pending", headers=headers)
    items = pending.json()["pending"]
    assert len(items) == 1
    assert items[0]["blocked_reason"] is not None

    approve = client.post(
        "/evolution/overlays/approve",
        json={"agent": "coding_agent", "content_hash": content_hash},
        headers=headers,
    )
    assert approve.status_code == 200
    result = approve.json()["approval"]
    assert result["status"] == "discarded"

    # No overlay version was created; agent tool policy is untouched.
    versions = client.get("/evolution/versions", params={"agent": "coding_agent"}, headers=headers)
    assert versions.json()["versions"] == []
    registry_agent = container.agent_registry.get("coding_agent")
    assert registry_agent is not None
    assert "unrestricted_shell" not in registry_agent.config.allowed_tools


@pytest.mark.asyncio
async def test_run_standalone_respects_gate_and_never_renices_when_skipped(
    settings_tmp,
) -> None:
    from services.evolution.dreamer import run_standalone

    # Evolution is disabled by default: the standalone worker must skip via the
    # gate before ever touching process priority.
    result = await run_standalone(settings_tmp, datetime(2026, 7, 3, 2, 30, tzinfo=UTC))
    assert result.status == "skipped"
    assert result.reason == "evolution disabled"


@pytest.mark.asyncio
async def test_overlay_approval_requires_exact_hash(settings_tmp) -> None:
    _seed_pending_candidate(settings_tmp, "Real candidate content.")
    database, _memory_unused = await _memory(settings_tmp)
    try:
        service = PromptOverlayApprovalService(settings_tmp, database)
        wrong = await service.approve(agent="coding_agent", content_hash="0" * 64)
        assert wrong.status == "discarded"
        assert wrong.reason == "no pending candidate matches that hash"
    finally:
        await database.close()
