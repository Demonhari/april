from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from april_common.audit import audit_logger_for_settings
from april_common.settings import AprilSettings
from services.evolution.rollouts import (
    CanaryContext,
    InvalidRolloutTransition,
    PromotionReadiness,
    RolloutBlocked,
    RolloutRecord,
    RolloutService,
    ShadowMetrics,
)
from services.evolution.versions import PromptOverlayManager
from services.jobs.registry import default_job_registry
from services.jobs.store import JobStore
from services.memory.database import Database
from services.memory.migrations import SCHEMA_VERSION, run_migrations
from services.permissions.approvals import ApprovalStore


class FakeShadowEvaluator:
    def __init__(
        self,
        metrics: ShadowMetrics,
        *,
        set_cancel: bool = False,
    ) -> None:
        self.metrics = metrics
        self.set_cancel = set_cancel

    async def evaluate(
        self,
        _rollout: RolloutRecord,
        *,
        cancellation_event: asyncio.Event | None = None,
    ) -> ShadowMetrics:
        if self.set_cancel and cancellation_event is not None:
            cancellation_event.set()
        return self.metrics


def _enabled(settings: AprilSettings, **overrides: Any) -> AprilSettings:
    evolution = settings.evolution.model_copy(
        update={
            "enabled": True,
            "rollout_enabled": True,
            "canary_enabled": True,
            "automatic_candidate_creation": False,
            "automatic_promotion": False,
            "rollout_shadow_min_samples": 2,
            "rollout_canary_min_samples": 2,
            "rollout_canary_fraction": 0.25,
            "rollout_canary_max_eligible_turns": 100,
            **overrides,
        }
    )
    return settings.model_copy(update={"evolution": evolution})


async def _database(settings: AprilSettings) -> Database:
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    return database


def _candidate(settings: AprilSettings, name: str = "candidate.overlay.txt") -> Path:
    path = settings.evolution_path / "candidates" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Prefer concise answers and explicitly preserve uncertainty.\n",
        encoding="utf-8",
    )
    return path


def _reviewed_case(settings: AprilSettings) -> None:
    path = settings.evolution_path / "evals" / "reviewed" / "reviewed-1.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "status: reviewed\nprompt: hello\nexpected_behavior: answer safely\n",
        encoding="utf-8",
    )


def _passing_metrics(samples: int = 2) -> ShadowMetrics:
    return ShadowMetrics(
        sample_count=samples,
        human_reviewed_sample_count=1,
        baseline_pass_count=samples,
        candidate_pass_count=samples,
        baseline_structured_valid_count=samples,
        candidate_structured_valid_count=samples,
        baseline_failure_count=0,
        candidate_failure_count=0,
        baseline_latency_ms=10,
        candidate_latency_ms=11,
        baseline_compared=True,
        human_reviewed_evidence_present=True,
    )


async def _create_and_shadow(
    settings: AprilSettings,
    database: Database,
    *,
    service: RolloutService | None = None,
) -> tuple[RolloutService, RolloutRecord]:
    _reviewed_case(settings)
    active = service or RolloutService(
        settings,
        database,
        audit=audit_logger_for_settings(settings),
    )
    record = await active.create(
        candidate_type="prompt_overlay",
        target_id="general_agent",
        candidate_id=f"candidate-{uuid.uuid4().hex}",
        candidate_artifact_path=_candidate(settings),
        minimum_samples=2,
    )
    record = await active.start_shadow(
        record.id,
        evaluator=FakeShadowEvaluator(_passing_metrics()),
    )
    assert record.state == "shadow_passed"
    return active, record


async def _approved(
    settings: AprilSettings,
    database: Database,
    service: RolloutService,
    record: RolloutRecord,
    stage: str,
) -> str:
    approvals = ApprovalStore(
        database,
        audit_logger_for_settings(settings),
        expiry_seconds=settings.permissions.approval_expiry_seconds,
    )
    approval_id = await service.request_approval(
        record.id,
        stage=stage,  # type: ignore[arg-type]
        approvals=approvals,
    )
    stored = await approvals.get(approval_id)
    await approvals.approve_exact(
        approval_id=approval_id,
        tool=stored.tool,
        args=stored.args,
        actor="local-user",
        request_id="approve-test",
    )
    return approval_id


async def _start_canary(
    settings: AprilSettings,
    database: Database,
    service: RolloutService,
    record: RolloutRecord,
) -> RolloutRecord:
    approval_id = await _approved(settings, database, service, record, "canary")
    return await service.start_canary(record.id, approval_id=approval_id)


async def _selected_request(
    service: RolloutService,
    record: RolloutRecord,
    *,
    start: int = 0,
) -> str:
    for index in range(start, start + 1000):
        request_id = f"request-{index}"
        selection = await service.select_prompt_canary(
            target_id=record.target_id,
            context=CanaryContext(
                stable_request_id=request_id,
                agent=record.target_id,
            ),
        )
        if selection.selected:
            assert selection.overlay_text is not None
            return request_id
    raise AssertionError("deterministic canary fraction selected no request")


async def _pass_canary(
    service: RolloutService,
    record: RolloutRecord,
) -> RolloutRecord:
    running = record
    start = 0
    while running.state == "canary_running":
        request_id = await _selected_request(service, running, start=start)
        start += int(request_id.rsplit("-", 1)[1]) + 1
        running = await service.record_canary_outcome(
            rollout_id=running.id,
            stable_request_id=request_id,
            outcome={
                "structured_output_valid": True,
                "success": True,
                "latency_ms": 10,
                "baseline_latency_ms": 10,
            },
        )
    assert running.state == "canary_passed"
    return running


@pytest.mark.asyncio
async def test_schema_21_rollout_tables_and_valid_state_transitions(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        row = await database.fetchone(
            "SELECT version FROM schema_migrations WHERE version = ?",
            (SCHEMA_VERSION,),
        )
        assert row is not None
        service, shadow = await _create_and_shadow(settings, database)
        running = await _start_canary(settings, database, service, shadow)
        assert running.state == "canary_running"
        assert running.canary_approval_id is not None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_invalid_transition_and_training_metrics_alone_are_rejected(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        service = RolloutService(settings, database)
        record = await service.create(
            candidate_type="prompt_overlay",
            target_id="general_agent",
            candidate_id="candidate-v1",
            candidate_artifact_path=_candidate(settings),
            minimum_samples=2,
        )
        with pytest.raises(InvalidRolloutTransition):
            await service.start_canary(record.id, approval_id="missing")
        metrics = _passing_metrics()
        metrics = ShadowMetrics(**{**metrics.safe_payload(), "training_metric_only": True})
        failed = await service.start_shadow(
            record.id,
            evaluator=FakeShadowEvaluator(metrics),
        )
        assert failed.state == "failed"
        assert failed.reason_code == "training_metrics_are_not_shadow_evidence"
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metrics", "reason"),
    [
        (
            ShadowMetrics(
                **{
                    **_passing_metrics(1).safe_payload(),
                    "sample_count": 1,
                    "baseline_pass_count": 1,
                    "candidate_pass_count": 1,
                }
            ),
            "shadow_minimum_samples_not_met",
        ),
        (
            ShadowMetrics(
                **{
                    **_passing_metrics().safe_payload(),
                    "candidate_pass_count": 1,
                }
            ),
            "shadow_pass_rate_regression",
        ),
    ],
)
async def test_shadow_minimum_samples_and_no_regression_gates(
    settings_tmp: AprilSettings,
    metrics: ShadowMetrics,
    reason: str,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        service = RolloutService(settings, database)
        record = await service.create(
            candidate_type="prompt_overlay",
            target_id="general_agent",
            candidate_id=f"candidate-{reason}",
            candidate_artifact_path=_candidate(settings, f"{reason}.overlay.txt"),
            minimum_samples=2,
        )
        result = await service.start_shadow(
            record.id,
            evaluator=FakeShadowEvaluator(metrics),
        )
        assert result.state == "failed"
        assert result.reason_code == reason
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_shadow_never_changes_active_overlay_and_cancellation_is_durable(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        manager = PromptOverlayManager(settings, database)
        baseline = await manager.apply_candidate(
            agent="general_agent",
            content="Baseline guidance remains exactly active for this test.",
            eval_score=1.0,
            baseline_score=1.0,
            approved=True,
        )
        assert baseline.status == "applied"
        before = await manager.active_overlay("general_agent")
        _reviewed_case(settings)
        service = RolloutService(settings, database)
        record = await service.create(
            candidate_type="prompt_overlay",
            target_id="general_agent",
            candidate_id="candidate-v2",
            candidate_artifact_path=_candidate(settings),
            minimum_samples=2,
        )
        event = asyncio.Event()
        cancelled = await service.start_shadow(
            record.id,
            evaluator=FakeShadowEvaluator(_passing_metrics(), set_cancel=True),
            cancellation_event=event,
        )
        assert cancelled.state == "cancelled"
        assert await manager.active_overlay("general_agent") == before
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_shadow_can_be_queued_as_restart_safe_durable_job(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        service = RolloutService(settings, database)
        record = await service.create(
            candidate_type="prompt_overlay",
            target_id="general_agent",
            candidate_id="queued-candidate",
            candidate_artifact_path=_candidate(settings),
            minimum_samples=2,
        )
        store = JobStore(
            database,
            default_job_registry(evolution_enabled=True),
        )
        pending, job = await service.queue_shadow(record.id, store=store)
        assert pending.state == "shadow_pending"
        assert pending.shadow_job_id == job.id
        assert job.job_type == "evolution_shadow"
        assert job.status.value == "queued"
        definition = store.registry.require("evolution_shadow")
        assert definition.restart_safe is True
        assert definition.idempotent is True
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_exact_level4_approval_and_concurrent_canary_start(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        service, shadow = await _create_and_shadow(settings, database)
        with pytest.raises(RolloutBlocked, match="approval_not_found"):
            await service.start_canary(shadow.id, approval_id="not-an-approval")
        pending = await service.require(shadow.id)
        assert pending.state == "canary_pending_approval"
        # A fresh record proves one exact approval can win only one concurrent
        # start; the other observes optimistic-concurrency failure.
        service2, shadow2 = await _create_and_shadow(
            settings,
            database,
            service=service,
        )
        approval_id = await _approved(settings, database, service2, shadow2, "canary")
        results = await asyncio.gather(
            service2.start_canary(shadow2.id, approval_id=approval_id),
            service2.start_canary(shadow2.id, approval_id=approval_id),
            return_exceptions=True,
        )
        assert sum(isinstance(item, RolloutRecord) for item in results) == 1
        assert sum(isinstance(item, Exception) for item in results) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_deterministic_canary_selection_and_high_risk_exclusion(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        service, shadow = await _create_and_shadow(settings, database)
        running = await _start_canary(settings, database, service, shadow)
        context = CanaryContext(stable_request_id="stable-17", agent="general_agent")
        first = await service.select_prompt_canary(
            target_id="general_agent",
            context=context,
        )
        second = await service.select_prompt_canary(
            target_id="general_agent",
            context=context,
        )
        assert first.selected == second.selected
        assert first.eligible == second.eligible
        high_risk = await service.select_prompt_canary(
            target_id="general_agent",
            context=CanaryContext(
                stable_request_id="high-risk",
                agent="general_agent",
                permission_level=4,
                risk_level="system_action",
                destructive=True,
            ),
        )
        assert high_risk.eligible is False
        assert high_risk.selected is False
        assert running.state == "canary_running"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_automatic_rollback_on_hard_failure_and_threshold_breach(
    settings_tmp: AprilSettings,
) -> None:
    for hard_failure in (True, False):
        settings = _enabled(
            settings_tmp,
            rollout_max_failure_rate=0.0,
        )
        # Each loop needs an isolated DB because candidate uniqueness is durable.
        settings = settings.model_copy(
            update={
                "memory": settings.memory.model_copy(
                    update={
                        "database_path": settings.home
                        / "data"
                        / f"rollout-{int(hard_failure)}.sqlite3"
                    }
                )
            }
        )
        database = await _database(settings)
        try:
            service, shadow = await _create_and_shadow(settings, database)
            running = await _start_canary(settings, database, service, shadow)
            request_id = await _selected_request(service, running)
            rolled = await service.record_canary_outcome(
                rollout_id=running.id,
                stable_request_id=request_id,
                outcome={
                    "structured_output_valid": not hard_failure,
                    "runtime_failure": hard_failure,
                    "hard_failure": hard_failure,
                    "tool_failure": not hard_failure,
                    "success": False,
                },
            )
            assert rolled.state == "rolled_back"
            assert rolled.reason_code in {
                "canary_hard_failure",
                "canary_failure_rate_threshold",
            }
        finally:
            await database.close()


@pytest.mark.asyncio
async def test_no_raw_content_can_be_stored_in_rollout_outcomes(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        service, shadow = await _create_and_shadow(settings, database)
        running = await _start_canary(settings, database, service, shadow)
        request_id = await _selected_request(service, running)
        with pytest.raises(ValueError, match="unsafe_rollout_outcome_key"):
            await service.record_canary_outcome(
                rollout_id=running.id,
                stable_request_id=request_id,
                outcome={"message": True},
            )
        row = await database.fetchone(
            """
            SELECT safe_outcome_json FROM evolution_rollout_assignments
            WHERE rollout_id = ? AND request_key_sha256 IS NOT NULL
              AND selected = 1
            """,
            (running.id,),
        )
        assert row is not None
        assert row["safe_outcome_json"] is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_interruption_reconciliation_before_and_after_pointer_publication(
    settings_tmp: AprilSettings,
) -> None:
    for phase in ("activation_prepared", "pointer_published"):
        settings = _enabled(settings_tmp)
        settings = settings.model_copy(
            update={
                "memory": settings.memory.model_copy(
                    update={"database_path": settings.home / "data" / f"{phase}.sqlite3"}
                )
            }
        )
        database = await _database(settings)
        try:

            async def fault(
                actual: str,
                _record: RolloutRecord,
                expected_phase: str = phase,
            ) -> None:
                if actual == expected_phase:
                    raise RuntimeError(f"interrupt-{expected_phase}")

            service = RolloutService(settings, database, fault_hook=fault)
            service, shadow = await _create_and_shadow(
                settings,
                database,
                service=service,
            )
            running = await _start_canary(settings, database, service, shadow)
            passed = await _pass_canary(service, running)
            approval_id = await _approved(
                settings,
                database,
                service,
                passed,
                "activation",
            )
            with pytest.raises(RuntimeError, match=f"interrupt-{phase}"):
                await service.promote(
                    passed.id,
                    approval_id=approval_id,
                    readiness=PromotionReadiness(True, True),
                )
            reconciliation = await RolloutService(
                settings,
                database,
                audit=audit_logger_for_settings(settings),
            ).reconcile_startup()
            assert reconciliation["reconciled_rollout_count"] == 1
            restored = await service.require(passed.id)
            assert restored.state == "rolled_back"
            assert not await service._candidate_is_active(restored)
        finally:
            await database.close()


@pytest.mark.asyncio
async def test_previous_artifact_restored_exactly_rollback_idempotent_and_audit_valid(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        audit = audit_logger_for_settings(settings)
        manager = PromptOverlayManager(settings, database, audit=audit)
        baseline = await manager.apply_candidate(
            agent="general_agent",
            content="Original baseline guidance that must be restored exactly.",
            eval_score=1.0,
            baseline_score=1.0,
            approved=True,
        )
        assert baseline.path is not None
        original = baseline.path.read_bytes()
        service, shadow = await _create_and_shadow(
            settings,
            database,
            service=RolloutService(settings, database, audit=audit),
        )
        running = await _start_canary(settings, database, service, shadow)
        passed = await _pass_canary(service, running)
        approval_id = await _approved(settings, database, service, passed, "activation")
        active = await service.promote(
            passed.id,
            approval_id=approval_id,
            readiness=PromotionReadiness(True, True),
        )
        assert active.state == "active"
        first = await service.rollback(active.id, reason_code="test_rollback")
        second = await service.rollback(active.id, reason_code="test_rollback")
        assert first.state == second.state == "rolled_back"
        assert await manager.active_overlay("general_agent") == original
        active_rows = await database.fetchall(
            "SELECT id FROM prompt_versions WHERE agent = ? AND active = 1",
            ("general_agent",),
        )
        assert len(active_rows) == 1
        assert audit.verify().status == "valid"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_newly_active_prompt_automatically_rolls_back_on_hard_failure(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        service, shadow = await _create_and_shadow(settings, database)
        running = await _start_canary(settings, database, service, shadow)
        passed = await _pass_canary(service, running)
        approval_id = await _approved(settings, database, service, passed, "activation")
        active = await service.promote(
            passed.id,
            approval_id=approval_id,
            readiness=PromotionReadiness(True, True),
        )
        manager = PromptOverlayManager(settings, database)
        request_id = "active-monitor-request"
        text = await manager.active_overlay_text(
            "general_agent",
            canary_context=CanaryContext(
                stable_request_id=request_id,
                agent="general_agent",
            ),
        )
        assert text is not None
        result = await service.record_canary_outcome_for_request(
            stable_request_id=request_id,
            outcome={
                "runtime_failure": True,
                "hard_failure": True,
                "success": False,
            },
        )
        assert result is not None
        assert result.state == "rolled_back"
        assert result.reason_code == "canary_hard_failure"
        assert active.state == "active"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_concurrent_promote_and_rollback_never_leave_two_active_artifacts(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        service, shadow = await _create_and_shadow(settings, database)
        running = await _start_canary(settings, database, service, shadow)
        passed = await _pass_canary(service, running)
        approval_id = await _approved(settings, database, service, passed, "activation")
        results = await asyncio.gather(
            service.promote(
                passed.id,
                approval_id=approval_id,
                readiness=PromotionReadiness(True, True),
            ),
            service.rollback(passed.id, reason_code="concurrent_rollback"),
            return_exceptions=True,
        )
        assert any(isinstance(item, RolloutRecord) for item in results)
        await service.reconcile_startup()
        rows = await database.fetchall(
            "SELECT id FROM prompt_versions WHERE agent = ? AND active = 1",
            ("general_agent",),
        )
        assert len(rows) <= 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_activation_cancellation_rolls_back(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        service, shadow = await _create_and_shadow(settings, database)
        running = await _start_canary(settings, database, service, shadow)
        passed = await _pass_canary(service, running)
        approval_id = await _approved(settings, database, service, passed, "activation")
        cancellation = asyncio.Event()
        cancellation.set()
        result = await service.promote(
            passed.id,
            approval_id=approval_id,
            readiness=PromotionReadiness(True, True),
            cancellation_event=cancellation,
        )
        assert result.state == "rolled_back"
        assert result.reason_code == "activation_cancelled"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_defaults_disarm_evolution_dreamer_canary_and_promotion(
    settings_tmp: AprilSettings,
) -> None:
    assert settings_tmp.evolution.enabled is False
    assert settings_tmp.evolution.rollout_enabled is False
    assert settings_tmp.evolution.canary_enabled is False
    assert settings_tmp.evolution.automatic_candidate_creation is False
    assert settings_tmp.evolution.automatic_promotion is False
    database = await _database(settings_tmp)
    try:
        with pytest.raises(RolloutBlocked, match="evolution_disabled"):
            await RolloutService(settings_tmp, database).create(
                candidate_type="prompt_overlay",
                target_id="general_agent",
                candidate_id="disabled-candidate",
                candidate_artifact_path=_candidate(settings_tmp),
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_lora_canary_is_explicitly_blocked_as_unsupported(
    settings_tmp: AprilSettings,
) -> None:
    settings = _enabled(settings_tmp)
    database = await _database(settings)
    try:
        adapter = settings.evolution_path / "adapters" / "candidates" / "candidate.gguf"
        adapter.parent.mkdir(parents=True, exist_ok=True)
        adapter.write_bytes(b"GGUF-test-adapter")
        _reviewed_case(settings)
        service = RolloutService(settings, database)
        record = await service.create(
            candidate_type="lora_adapter",
            target_id="april-brain",
            candidate_id="adapter-v1",
            candidate_artifact_path=adapter,
            minimum_samples=2,
        )
        shadow = await service.start_shadow(
            record.id,
            evaluator=FakeShadowEvaluator(_passing_metrics()),
        )
        approval_id = await _approved(settings, database, service, shadow, "canary")
        with pytest.raises(RolloutBlocked, match="lora_canary_unsupported"):
            await service.start_canary(shadow.id, approval_id=approval_id)
    finally:
        await database.close()


def test_rollout_metrics_json_contains_no_conversation_fields() -> None:
    payload = json.dumps(_passing_metrics().safe_payload(), sort_keys=True)
    assert "prompt" not in payload
    assert "message" not in payload
    assert "conversation" not in payload
    assert "tool_output" not in payload
