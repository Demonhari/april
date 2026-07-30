# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import uuid
from typing import Any

from agents.schemas import AgentResult
from april_common.errors import PermissionDeniedError
from april_common.time import parse_utc_iso, utc_now
from services.brain.memory_policy import build_agent_memory_context
from services.evolution.feedback_eval import stage_feedback_eval_case


class ApprovalFlow:
    async def run_agent(
        self,
        *,
        agent_id: str,
        message: str,
        conversation_id: str | None = None,
        request_id: str | None = None,
        actor: str = "local-user",
        project_id: str | None = None,
        repo_path: str | None = None,
    ) -> AgentResult:
        active_request_id = request_id or str(uuid.uuid4())
        agent = self.agent_registry.get(agent_id)
        if agent is None:
            raise PermissionDeniedError("Unknown agent.", {"agent": agent_id})
        # Direct specialist runs do not carry a routed, deterministic
        # low-risk eligibility decision, so they always use the active
        # baseline/full-activation overlay and are excluded from canary.
        agent = await self.apply_prompt_overlay(agent)
        agent, run_metadata = await self._effective_agent(agent)
        project = await self._resolve_project(project_id=project_id, repo_path=repo_path)
        if self._agent_requires_project(agent_id) and project is None:
            raise PermissionDeniedError(
                "This agent requires a selected local project.",
                {"agent": agent_id},
            )
        active_conversation_id = conversation_id or await self.memory.create_conversation(
            project_id=project.id if project else None,
            actor=actor,
        )
        if conversation_id is not None:
            await self.memory.ensure_conversation(
                active_conversation_id,
                project_id=project.id if project else None,
                actor=actor,
            )
        prepared_context = await self.conversation_context.prepare(
            conversation_id=active_conversation_id,
            request_id=active_request_id,
        )
        raw_history = prepared_context.recent_messages
        await self.memory.add_message(active_conversation_id, "user", message)
        memory_context = await build_agent_memory_context(
            policy=agent.config.memory_access_policy,
            history=raw_history,
            memory_retriever=self.memory_retriever,
            memory_queries=[],
            intent="direct_agent_run",
            message=message,
            project=project,
            conversation_summary=prepared_context.summary,
            budgets=self.settings.conversation_context,
            user_model_path=self.settings.evolution_path / "user_model.md",
        )
        run_metadata.update(prepared_context.diagnostics())
        run_metadata["context_category_character_usage"] = dict(
            memory_context.category_character_usage
        )
        run_metadata["context_category_truncated"] = dict(memory_context.category_truncated)
        context_sections, _context_citations = self._memory_context_sections(memory_context)
        if memory_context.conversation_summary:
            context_sections.insert(0, memory_context.conversation_summary)
        context = await self.tool_executor.context(
            request_id=active_request_id,
            conversation_id=active_conversation_id,
            actor=actor,
            agent_id=agent.name,
            project_id=project.id if project else None,
            source="chat",
        )
        result = await self.structured_loop.run(
            agent=agent,
            message=message,
            context=context,
            request_id=active_request_id,
            history=memory_context.history,
            context_sections=context_sections,
            run_metadata=run_metadata,
        )
        if result.status != "pending_approval":
            await self.memory.add_message(active_conversation_id, "assistant", result.final_message)
        return result.model_copy(
            update={
                "warnings": [*prepared_context.warnings, *result.warnings],
                "metadata": {**run_metadata, **result.metadata},
            }
        )

    async def approve_tool(
        self,
        *,
        approval_id: str,
        actor: str,
        request_id: str,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        approval = await self.approvals.get(approval_id)
        suspended = await self.memory.get_suspended_agent_run_by_approval(approval_id)
        belongs_to_agent_run = approval.metadata.get("agent_run_id") is not None
        if belongs_to_agent_run and suspended is None:
            raise PermissionDeniedError("Suspended agent run is no longer available.")
        if suspended is None:
            outcome = await self.tool_executor.execute_approved(
                approval_id=approval_id,
                actor=actor,
                request_id=request_id,
                tool=tool,
                args=args,
            )
            route_key = approval.metadata.get("route_key")
            if isinstance(route_key, str):
                tool_ok = outcome.result is not None and outcome.result.ok
                await self.routing_reliability.mark_latest_route_outcome(
                    route_key=route_key,
                    approval_outcome="approved",
                    tool_outcome="success" if tool_ok else "failed",
                    coding_test_outcome=(
                        ("passed" if tool_ok else "failed")
                        if approval.tool == "test_runner"
                        else None
                    ),
                    final_status="ok" if tool_ok else "error",
                )
            return {"status": outcome.status, "result": outcome.result}
        if suspended.status != "suspended":
            raise PermissionDeniedError(
                "Suspended agent run is not resumable.",
                {"status": suspended.status},
            )
        if approval.status != "pending":
            raise PermissionDeniedError("Approval is not pending.", {"status": approval.status})
        if parse_utc_iso(approval.expires_at) < utc_now():
            await self.approvals.expire_pending(
                approval_id=approval_id,
                actor=actor,
                request_id=request_id,
            )
            if suspended is not None:
                await self.routing_reliability.mark_approval_outcome(
                    agent_run_id=suspended.agent_run_id,
                    outcome="expired",
                    final_status="expired",
                )
            raise PermissionDeniedError("Approval has expired.")
        if await self.memory.get_conversation(suspended.conversation_id) is None:
            await self.memory.mark_agent_run_failed(approval_id=approval_id)
            raise PermissionDeniedError("Conversation for suspended agent run no longer exists.")
        if (
            suspended.project_id is not None
            and await self.memory.get_project(suspended.project_id) is None
        ):
            await self.memory.mark_agent_run_failed(approval_id=approval_id)
            raise PermissionDeniedError("Project for suspended agent run no longer exists.")
        metadata_project_id = approval.metadata.get("project_id")
        if (
            metadata_project_id is not None
            and await self.memory.get_project(str(metadata_project_id)) is None
        ):
            await self.memory.mark_agent_run_failed(approval_id=approval_id)
            raise PermissionDeniedError("Project for suspended agent run no longer exists.")
        outcome = await self.tool_executor.execute_approved(
            approval_id=approval_id,
            actor=actor,
            request_id=request_id,
            tool=tool,
            args=args,
        )
        if outcome.result is None:
            await self.memory.mark_agent_run_failed(approval_id=approval_id)
            return {"status": outcome.status, "result": outcome.result}
        if not outcome.result.ok:
            await self.structured_loop.fail_suspended(
                suspended=suspended,
                tool_result=outcome.result,
            )
            return {"status": outcome.status, "result": outcome.result}
        agent = self.agent_registry.get(suspended.agent)
        if agent is None:
            await self.memory.mark_agent_run_failed(approval_id=approval_id)
            raise PermissionDeniedError("Suspended agent no longer exists.")
        context = await self.tool_executor.context(
            request_id=request_id,
            conversation_id=suspended.conversation_id,
            actor=actor,
            agent_id=suspended.agent,
            project_id=suspended.project_id,
            approval_id=approval_id,
            source="approval",
        )
        result = await self.structured_loop.resume(
            suspended=suspended,
            agent=agent,
            context=context,
            tool_result=outcome.result,
            request_id=request_id,
        )
        if result.status != "pending_approval":
            await self.memory.add_message(
                suspended.conversation_id, "assistant", result.final_message
            )
        await self.routing_reliability.mark_approval_outcome(
            agent_run_id=suspended.agent_run_id,
            outcome="approved",
            final_status=result.status,
        )
        return {"status": "resumed", "result": result.model_dump()}

    async def deny_tool(
        self,
        *,
        approval_id: str,
        actor: str,
        request_id: str,
    ) -> dict[str, Any]:
        approval = await self.approvals.get(approval_id)
        suspended = await self.memory.get_suspended_agent_run_by_approval(approval_id)
        belongs_to_agent_run = approval.metadata.get("agent_run_id") is not None
        if belongs_to_agent_run and suspended is None:
            raise PermissionDeniedError("Suspended agent run is no longer available.")
        await self.approvals.deny(
            approval_id=approval_id,
            actor=actor,
            request_id=request_id,
        )
        await self._record_denial_feedback(approval, suspended)
        if suspended is None:
            route_key = approval.metadata.get("route_key")
            if isinstance(route_key, str):
                await self.routing_reliability.mark_latest_route_outcome(
                    route_key=route_key,
                    approval_outcome="denied",
                    final_status="denied",
                )
                if self.overlay_manager is not None:
                    from services.evolution.rollouts import RolloutService

                    await RolloutService(
                        self.settings,
                        self.memory.database,
                        audit=self.approvals.audit,
                    ).record_signal_for_agent_run(
                        agent_run_id=suspended.agent_run_id,
                        signal="approval_denied",
                    )
            return {"status": "denied", "approval_id": approval_id}
        result = AgentResult(
            status="error",
            final_message="Approval denied. The suspended agent run was stopped.",
            conversation_id=suspended.conversation_id,
        )
        await self.memory.record_conversation_event(
            conversation_id=suspended.conversation_id,
            event_type="agent_denied",
            payload={"approval_id": approval_id, "run_id": suspended.agent_run_id},
        )
        return {"status": "denied", "approval_id": approval_id, "result": result.model_dump()}

    async def _record_denial_feedback(self, approval: Any, suspended: Any) -> None:
        """A denied approval is an explicit negative signal about the proposal.

        Recorded as a feedback event so evolution/scorecards can learn from it;
        failures are swallowed because denial itself must always succeed.
        """
        try:
            if suspended is not None:
                await self.routing_reliability.mark_approval_outcome(
                    agent_run_id=suspended.agent_run_id,
                    outcome="denied",
                    final_status="denied",
                )
            record = await self.memory.record_feedback_event(
                rating="bad",
                reason=f"approval_denied: {approval.tool}",
                conversation_id=(suspended.conversation_id if suspended is not None else None),
                agent_run_id=(suspended.agent_run_id if suspended is not None else None),
            )
            self.approvals.audit.write(
                {
                    "event_type": "feedback_recorded",
                    "actor": "local-user",
                    "rating": record.rating,
                    "kind": "approval_denied",
                    "reason_length": len(record.reason or ""),
                    "agent_run_bound": record.agent_run_id is not None,
                }
            )
            await stage_feedback_eval_case(
                self.settings,
                self.memory,
                record,
                kind="approval_denied",
                audit=self.approvals.audit,
            )
        except Exception:
            return
