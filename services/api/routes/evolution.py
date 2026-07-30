from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

from services.api.dependencies import ApiContainer
from services.api.schemas import (
    AdapterActivateRequest,
    AdapterRollbackRequest,
    DatasetExportRequest,
    EvalPromoteRequest,
    EvalRejectRequest,
    EvolutionRollbackRequest,
    FeedbackRequest,
    OverlayApprovalRequest,
    PlaybookResumeRequest,
    PlaybookRunRequest,
)
from services.evolution.adapters import AdapterLifecycleManager
from services.evolution.approval import PromptOverlayApprovalService
from services.evolution.dataset_export import export_finetune_dataset
from services.evolution.dreamer import latest_report
from services.evolution.eval_review import (
    EvalReviewError,
    get_pending_case,
    list_pending_cases,
    promote_pending_case,
    reject_pending_case,
)
from services.evolution.feedback_eval import stage_feedback_eval_case
from services.evolution.inspect import (
    evolution_history,
    evolution_status,
    overlay_diff,
    set_evolution_kill_switch,
)
from services.evolution.playbook_miner import mine_playbook_candidates
from services.evolution.rollouts import RolloutService
from services.evolution.versions import PromptOverlayManager
from services.evolution.write_guard import EvolutionWriteGuard
from skills.playbooks import (
    PlaybookAdoptionService,
    PlaybookDefinition,
    PlaybookLoader,
    PlaybookRunner,
)


def register_evolution_routes(
    app: FastAPI,
    authorized: Callable[..., Any],
) -> None:
    @app.post("/feedback")
    async def feedback(
        request: FeedbackRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        conversation_id = request.conversation_id
        session_id: str | None = None
        if conversation_id is None:
            # Bind to the active session's conversation when one exists.
            session = await active.memory.latest_open_session()
            if session is not None:
                session_id = session.id
                conversation_id = session.conversation_id
        agent_run_id = request.agent_run_id
        if agent_run_id is None:
            agent_run_id = await active.memory.latest_agent_run_id(conversation_id=conversation_id)
        record = await active.memory.record_feedback_event(
            rating=request.rating,
            reason=request.reason,
            session_id=session_id,
            conversation_id=conversation_id,
            agent_run_id=agent_run_id,
        )
        if record.rating == "bad" and record.agent_run_id is not None:
            await active.orchestrator.routing_reliability.mark_negative_feedback(
                agent_run_id=record.agent_run_id
            )
            with contextlib.suppress(Exception):
                await RolloutService(
                    active.settings,
                    active.database,
                    audit=active.approvals.audit,
                ).record_signal_for_agent_run(
                    agent_run_id=record.agent_run_id,
                    signal="negative_feedback",
                )
        active.approvals.audit.write(
            {
                "event_type": "feedback_recorded",
                "request_id": request_id,
                "actor": "local-user",
                "rating": record.rating,
                "reason_length": len(record.reason or ""),
                "agent_run_bound": record.agent_run_id is not None,
            }
        )
        if record.rating == "bad":
            # Stage a reviewable pending eval case; never let staging failures
            # break feedback recording itself.
            with contextlib.suppress(Exception):
                await stage_feedback_eval_case(
                    active.settings,
                    active.memory,
                    record,
                    kind="explicit_feedback",
                    audit=active.approvals.audit,
                )
        return {"feedback": record.model_dump()}

    @app.get("/playbooks")
    async def playbooks(active: ApiContainer = Depends(authorized)) -> object:
        loader = PlaybookLoader(active.settings.playbooks_path)
        return {"playbooks": [playbook.model_dump() for playbook in loader.list()]}

    @app.post("/playbooks/adopt")
    async def playbook_adopt(
        request: PlaybookDefinition,
        approval_id: str | None = None,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        # Level 3+ playbooks require an exact-action adoption approval: the
        # first call returns pending_approval; re-posting the identical
        # definition with ?approval_id=... completes adoption.
        adoption = PlaybookAdoptionService(
            loader=PlaybookLoader(active.settings.playbooks_path),
            tool_registry=active.tool_registry,
            approvals=active.approvals,
            memory=active.memory,
            approval_required_at=active.permission_engine.approval_required_at,
        )
        return await adoption.adopt(
            request,
            actor="local-user",
            request_id=x_request_id or str(uuid.uuid4()),
            approval_id=approval_id,
        )

    @app.post("/playbooks/mine")
    async def playbook_mine(
        support_threshold: int = 3,
        lookback_days: int = 14,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        report = await mine_playbook_candidates(
            active.memory,
            active.settings,
            guard=EvolutionWriteGuard(active.settings, audit=active.approvals.audit),
            support_threshold=max(2, support_threshold),
            lookback_days=max(1, lookback_days),
        )
        return {"mine": report.to_payload()}

    @app.post("/playbooks/{playbook_id}/run")
    async def playbook_run(
        playbook_id: str,
        request: PlaybookRunRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        loader = PlaybookLoader(active.settings.playbooks_path)
        playbook = loader.get(playbook_id)
        if playbook is None:
            raise HTTPException(status_code=404, detail="playbook not found")
        async with active.require_session_manager().interaction(request.conversation_id):
            result = await PlaybookRunner(active.tool_executor, memory=active.memory).run(
                playbook,
                conversation_id=request.conversation_id,
                project_id=request.project_id,
            )
        return {"run": asdict(result)}

    @app.get("/playbooks/{playbook_id}/runs")
    async def playbook_runs(
        playbook_id: str,
        limit: int = 50,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        runs = await active.memory.list_playbook_runs(playbook_id=playbook_id, limit=limit)
        return {"runs": runs}

    @app.post("/playbooks/runs/{run_id}/resume")
    async def playbook_resume(
        run_id: str,
        request: PlaybookResumeRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        row = await active.memory.get_playbook_run(run_id)
        conversation_id = (
            str(row["conversation_id"])
            if row is not None and row["conversation_id"] is not None
            else None
        )
        async with active.require_session_manager().interaction(conversation_id):
            result = await PlaybookRunner(active.tool_executor, memory=active.memory).resume(
                run_id, approval_id=request.approval_id
            )
        return {"run": asdict(result)}

    @app.get("/evolution/versions")
    async def evolution_versions(
        agent: str | None = None,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = PromptOverlayManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        return {"versions": await manager.versions(agent=agent)}

    @app.post("/evolution/rollback")
    async def evolution_rollback(
        request: EvolutionRollbackRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = PromptOverlayManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        result = await manager.rollback(agent=request.agent, version=request.version)
        payload = asdict(result)
        if payload.get("path") is not None:
            payload["path"] = str(payload["path"])
        return {"rollback": payload}

    @app.get("/evolution/adapters")
    async def evolution_adapters(
        model_id: str | None = None,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = AdapterLifecycleManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        return {"adapters": await manager.list(model_id=model_id)}

    @app.post("/evolution/adapters/activate")
    async def evolution_adapters_activate(
        request: AdapterActivateRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = AdapterLifecycleManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        result = await manager.activate(
            model_id=request.model_id,
            adapter_path=Path(request.adapter_path),
            evidence_path=(Path(request.evidence_path) if request.evidence_path else None),
            verification_report_path=(
                Path(request.verification_report_path) if request.verification_report_path else None
            ),
        )
        return {"activation": result.to_payload()}

    @app.post("/evolution/adapters/rollback")
    async def evolution_adapters_rollback(
        request: AdapterRollbackRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = AdapterLifecycleManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        result = await manager.rollback(model_id=request.model_id, version=request.version)
        return {"rollback": result.to_payload()}

    @app.get("/evolution/report/latest")
    async def evolution_report_latest(active: ApiContainer = Depends(authorized)) -> object:
        return {"report": latest_report(active.settings)}

    @app.get("/evolution/status")
    async def evolution_status_endpoint(active: ApiContainer = Depends(authorized)) -> object:
        status = await evolution_status(active.settings, active.database)
        status["scheduler_running"] = active.scheduler.running if active.scheduler else False
        return {"status": status}

    @app.get("/evolution/history")
    async def evolution_history_endpoint(
        limit: int = 20,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return {"runs": await evolution_history(active.database, limit=limit)}

    @app.get("/evolution/diff")
    async def evolution_diff_endpoint(
        agent: str,
        from_version: int | None = None,
        to_version: int | None = None,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return await overlay_diff(
            active.settings,
            active.database,
            agent=agent,
            from_version=from_version,
            to_version=to_version,
        )

    @app.post("/evolution/off")
    async def evolution_off(
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        result = set_evolution_kill_switch(active.settings, disabled=True)
        active.approvals.audit.write(
            {
                "event_type": "evolution_kill_switch_set",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "outcome": "disabled",
            }
        )
        return result

    @app.post("/evolution/on")
    async def evolution_on(
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        result = set_evolution_kill_switch(active.settings, disabled=False)
        active.approvals.audit.write(
            {
                "event_type": "evolution_kill_switch_set",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "outcome": "cleared",
            }
        )
        return result

    @app.post("/evolution/dataset/export")
    async def evolution_dataset_export(
        request: DatasetExportRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        result = await export_finetune_dataset(
            active.memory,
            active.settings,
            dataset_name=request.name,
        )
        active.approvals.audit.write(
            {
                "event_type": "evolution_dataset_exported",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "outcome": "written",
            }
        )
        return {"export": result.to_payload()}

    @app.get("/evolution/overlays/pending")
    async def evolution_overlays_pending(
        active: ApiContainer = Depends(authorized),
    ) -> object:
        service = PromptOverlayApprovalService(
            active.settings,
            active.database,
            audit=active.approvals.audit,
            runtime_client=active.runtime_client,
        )
        return {"pending": [item.to_payload() for item in await service.list_pending()]}

    @app.post("/evolution/overlays/approve")
    async def evolution_overlays_approve(
        request: OverlayApprovalRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        service = PromptOverlayApprovalService(
            active.settings,
            active.database,
            audit=active.approvals.audit,
            runtime_client=active.runtime_client,
        )
        result = await service.approve(agent=request.agent, content_hash=request.content_hash)
        payload = asdict(result)
        if payload.get("path") is not None:
            payload["path"] = str(payload["path"])
        return {"approval": payload}

    @app.get("/evolution/evals/pending")
    async def evolution_evals_pending(active: ApiContainer = Depends(authorized)) -> object:
        return {"pending": list_pending_cases(active.settings)}

    @app.get("/evolution/evals/pending/{case_id}")
    async def evolution_evals_pending_case(
        case_id: str,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        try:
            case = get_pending_case(active.settings, case_id)
        except EvalReviewError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if case is None:
            raise HTTPException(status_code=404, detail="pending eval case not found")
        return {"case": case}

    @app.post("/evolution/evals/promote")
    async def evolution_evals_promote(
        request: EvalPromoteRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        try:
            result = promote_pending_case(
                active.settings,
                request.case_id,
                expected_behavior=request.expected_behavior,
                audit=active.approvals.audit,
            )
        except EvalReviewError as exc:
            status_code = 404 if "unknown" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return {"promoted": result}

    @app.post("/evolution/evals/reject")
    async def evolution_evals_reject(
        request: EvalRejectRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        try:
            result = reject_pending_case(
                active.settings,
                request.case_id,
                reason=request.reason,
                audit=active.approvals.audit,
            )
        except EvalReviewError as exc:
            status_code = 404 if "unknown" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return {"rejected": result}
