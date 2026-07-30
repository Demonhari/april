# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import uuid
from typing import Any

from agents.schemas import LocalCitation
from april_common.errors import PermissionDeniedError
from services.brain.execution import PreparedTurn
from services.brain.memory_policy import build_agent_memory_context
from services.brain.planner import task_plan_from_decision
from services.brain.reasoning_resolver import resolve_reasoning_model
from services.brain.schemas import (
    RouteResult,
    RouteSource,
)


class ContextFlow:
    async def _prepare_turn(
        self,
        message: str,
        *,
        conversation_id: str | None,
        request_id: str | None,
        actor: str,
        project_id: str | None,
        repo_path: str | None,
        structured_specialists: bool = False,
        mode: str = "standard",
    ) -> PreparedTurn:
        active_request_id = request_id or str(uuid.uuid4())
        project = await self._resolve_project(project_id=project_id, repo_path=repo_path)
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
        router_history = self._history_with_summary(
            active_conversation_id,
            prepared_context.summary,
            raw_history,
        )
        await self.memory.add_message(active_conversation_id, "user", message)
        route_result_method = getattr(self.brain_router, "route_result", None)
        if route_result_method is None:
            decision = await self.brain_router.route(
                message,
                request_id=active_request_id,
                history=router_history,
            )
            route_result = RouteResult(
                decision=decision,
                route_source=RouteSource(decision.routing_method),
                raw_model_confidence=(
                    decision.confidence if decision.routing_method != "fallback" else None
                ),
                effective_confidence=decision.confidence,
                confidence_source="legacy_router",
                repair_used=decision.routing_method == "model_repair",
            )
        else:
            route_result = await route_result_method(
                message,
                request_id=active_request_id,
                history=router_history,
            )
            decision = route_result.decision
        route_result = await self.routing_reliability.calibrate(route_result)
        await self.memory.record_conversation_event(
            conversation_id=active_conversation_id,
            event_type="brain_decision",
            payload={
                "intent": decision.intent[:64],
                "agent": decision.agent,
                "route_source": route_result.route_source.value,
                "matched_rule": route_result.matched_rule,
                "fallback_reason": route_result.fallback_reason,
                "raw_model_confidence": route_result.raw_model_confidence,
                "historical_reliability": route_result.historical_reliability,
                "effective_confidence": route_result.effective_confidence,
                "reliability_sample_count": route_result.reliability_sample_count,
                "confidence_source": route_result.confidence_source,
                "normalized_tool_classes": sorted(
                    {call.tool[:64] for call in decision.planned_tool_calls}
                    | {tool[:64] for tool in decision.tools_needed}
                ),
            },
        )
        agent = self.agent_registry.get(decision.agent)
        if agent is None:
            raise PermissionDeniedError(
                "Unknown agent selected by brain.", {"agent": decision.agent}
            )
        predicted_selection = self.intelligence_ladder.select(
            message=message,
            decision=decision,
            mode=mode,
            effective_confidence=route_result.effective_confidence,
        )
        agent = await self.apply_prompt_overlay(
            agent,
            request_id=active_request_id,
            decision=decision,
            mode=mode,
            high_risk_reasoning=(predicted_selection.high_stakes or predicted_selection.rung >= 2),
        )
        model_id = agent.model_id or decision.model_id
        run_metadata: dict[str, Any] = {
            **prepared_context.diagnostics(),
            "route_source": route_result.route_source.value,
            "matched_rule": route_result.matched_rule,
            "fallback_reason": route_result.fallback_reason,
            "raw_routing_confidence": route_result.raw_model_confidence,
            "historical_routing_reliability": route_result.historical_reliability,
            "effective_routing_confidence": route_result.effective_confidence,
            "routing_reliability_sample_count": route_result.reliability_sample_count,
            "routing_confidence_source": route_result.confidence_source,
        }
        if self.overlay_manager is not None:
            from services.evolution.rollouts import RolloutService

            rollout_id = await RolloutService(
                self.settings,
                self.memory.database,
                audit=self.approvals.audit,
            ).rollout_for_request(active_request_id)
            if rollout_id is not None:
                run_metadata["rollout_id"] = rollout_id
        if agent.model_id is not None and decision.model_id != agent.model_id:
            decision = decision.model_copy(update={"model_id": agent.model_id})
            route_result = route_result.model_copy(update={"decision": decision})
        if agent.name == "reasoning_agent":
            resolution = await resolve_reasoning_model(
                runtime_client=self.runtime_client,
                fallback_model_id=model_id,
            )
            model_id = resolution.model_id
            run_metadata["model_resolution"] = resolution.metadata()
        task_plan = task_plan_from_decision(
            decision,
            conversation_id=active_conversation_id,
            request_id=active_request_id,
        ).model_copy(update={"status": "running"})
        await self.memory.create_task_plan(task_plan)
        memory_context = await build_agent_memory_context(
            policy=agent.config.memory_access_policy,
            history=raw_history,
            memory_retriever=self.memory_retriever,
            memory_queries=decision.memory_queries,
            intent=decision.intent,
            message=message,
            project=project,
            conversation_summary=prepared_context.summary,
            budgets=self.settings.conversation_context,
            user_model_path=self.settings.evolution_path / "user_model.md",
        )
        run_metadata["context_category_character_usage"] = dict(
            memory_context.category_character_usage
        )
        run_metadata["context_category_truncated"] = dict(memory_context.category_truncated)
        context_sections, _context_citations = self._memory_context_sections(memory_context)
        if memory_context.conversation_summary:
            context_sections.insert(0, memory_context.conversation_summary)

        if self._requires_project(decision) and project is None:
            return PreparedTurn(
                request_id=active_request_id,
                conversation_id=active_conversation_id,
                decision=decision,
                route_result=route_result,
                agent_name=agent.name,
                model_id=model_id,
                messages=[],
                final_message=(
                    "This request needs a selected local project. Add one with "
                    "`april project add PATH`, then pass its project ID or repo path."
                ),
                warnings=[
                    *prepared_context.warnings,
                    "No project was selected for repository analysis.",
                ],
                project_id=None,
                actor=actor,
                history=memory_context.history,
                context_sections=context_sections,
                task_plan_id=task_plan.id,
                run_metadata=run_metadata,
            )

        if (
            structured_specialists
            and route_result.route_source is not RouteSource.DETERMINISTIC
            and self._uses_structured_loop(agent.name, decision)
        ):
            return PreparedTurn(
                request_id=active_request_id,
                conversation_id=active_conversation_id,
                decision=decision,
                route_result=route_result,
                agent_name=agent.name,
                model_id=model_id,
                messages=[],
                project_id=project.id if project else None,
                actor=actor,
                history=memory_context.history,
                context_sections=context_sections,
                structured_agent=True,
                warnings=list(prepared_context.warnings),
                task_plan_id=task_plan.id,
                run_metadata=run_metadata,
            )

        if decision.intent == "code_modification" and project is not None:
            return await self._prepare_code_modification(
                message=message,
                decision=decision,
                route_result=route_result,
                agent_name=agent.name,
                agent_prompt=agent.system_prompt,
                model_id=model_id,
                project=project,
                request_id=active_request_id,
                conversation_id=active_conversation_id,
                actor=actor,
                memory_context=memory_context,
                task_plan_id=task_plan.id,
            )

        planned_calls = self._planned_tool_calls(decision, message=message, project=project)
        if decision.intent in {"approval_command", "rejection_command"}:
            approval_id = str(planned_calls[0].args["approval_id"])
            if decision.intent == "approval_command":
                approval_result = await self.approve_tool(
                    approval_id=approval_id,
                    actor=actor,
                    request_id=active_request_id,
                )
                nested = approval_result.get("result")
                final_message = (
                    str(nested.get("final_message"))
                    if isinstance(nested, dict) and nested.get("final_message")
                    else f"Approval {approval_id} was consumed once."
                )
            else:
                await self.deny_tool(
                    approval_id=approval_id,
                    actor=actor,
                    request_id=active_request_id,
                )
                final_message = f"Approval {approval_id} was rejected."
            return PreparedTurn(
                request_id=active_request_id,
                conversation_id=active_conversation_id,
                decision=decision,
                route_result=route_result,
                agent_name=agent.name,
                model_id=model_id,
                messages=[],
                final_message=final_message,
                final_status="ok",
                project_id=project.id if project else None,
                actor=actor,
                history=memory_context.history,
                context_sections=context_sections,
                task_plan_id=task_plan.id,
                run_metadata=run_metadata,
            )
        tool_outputs: list[str] = []
        tool_failures: list[str] = []
        citations: list[LocalCitation] = []
        pending_approval: dict[str, Any] | None = None
        warnings: list[str] = list(prepared_context.warnings)
        memory_write_message: str | None = None
        for planned in planned_calls[: self.settings.permissions.maximum_agent_tool_iterations]:
            missing = self._missing_required_args(planned)
            if missing:
                warnings.append(
                    f"Tool {planned.tool} was not run because required arguments are missing: "
                    + ", ".join(missing)
                )
                continue
            context = await self.tool_executor.context(
                request_id=active_request_id,
                conversation_id=active_conversation_id,
                actor=actor,
                agent_id=agent.name,
                project_id=project.id
                if project
                else (str(planned.args["project_id"]) if planned.args.get("project_id") else None),
                source="orchestrator",
            )
            outcome = await self.tool_executor.request_or_execute(
                tool=planned.tool,
                args=planned.args,
                context=context,
                model_permission_level=decision.permission_level,
                model_risk_level=decision.risk_level,
                approval_metadata={"route_key": route_result.route_key},
            )
            if outcome.approval is not None:
                pending_approval = outcome.approval.model_dump()
                break
            tool_result = outcome.result
            if tool_result is None:
                continue
            if not tool_result.ok:
                tool_failures.append(
                    f"{planned.tool}: {tool_result.stderr or 'tool execution failed'}"
                )
            if tool_result.stdout:
                tool_outputs.append(f"{planned.tool}:\n{tool_result.stdout}")
            if planned.tool == "remember_memory" and tool_result.ok:
                memory_write_message = tool_result.stdout
            if planned.tool == "read_file" and tool_result.ok:
                citations.append(
                    LocalCitation(
                        path=tool_result.data.get("path", ""),
                        start_line=tool_result.data.get("start_line"),
                        end_line=tool_result.data.get("end_line"),
                    )
                )

        if decision.intent == "memory_write" and pending_approval is None:
            return PreparedTurn(
                request_id=active_request_id,
                conversation_id=active_conversation_id,
                decision=decision,
                route_result=route_result,
                agent_name=agent.name,
                model_id=model_id,
                messages=[],
                citations=citations,
                final_message=memory_write_message or "Stored memory.",
                final_status="ok",
                warnings=warnings,
                project_id=project.id if project else None,
                actor=actor,
                history=memory_context.history,
                context_sections=context_sections,
                task_plan_id=task_plan.id,
                run_metadata=run_metadata,
            )

        if route_result.route_source is RouteSource.DETERMINISTIC:
            if pending_approval is not None:
                return PreparedTurn(
                    request_id=active_request_id,
                    conversation_id=active_conversation_id,
                    decision=decision,
                    route_result=route_result,
                    agent_name=agent.name,
                    model_id=model_id,
                    messages=[],
                    citations=citations,
                    pending_approval=pending_approval,
                    final_message=(
                        "The exact local action is paused for one-time approval.\n"
                        f"Approval required: {pending_approval['approval_id']}"
                    ),
                    warnings=warnings,
                    project_id=project.id if project else None,
                    actor=actor,
                    history=memory_context.history,
                    context_sections=context_sections,
                    task_plan_id=task_plan.id,
                    run_metadata=run_metadata,
                )
            if planned_calls and decision.intent != "patch_proposal":
                final_message = "\n\n".join(tool_outputs or ["The local operation completed."])
                if tool_failures:
                    final_message = "\n\n".join(tool_failures)
                return PreparedTurn(
                    request_id=active_request_id,
                    conversation_id=active_conversation_id,
                    decision=decision,
                    route_result=route_result,
                    agent_name=agent.name,
                    model_id=model_id,
                    messages=[],
                    citations=citations,
                    final_message=final_message,
                    final_status="error" if tool_failures else "ok",
                    warnings=warnings,
                    project_id=project.id if project else None,
                    actor=actor,
                    history=memory_context.history,
                    context_sections=context_sections,
                    task_plan_id=task_plan.id,
                    run_metadata=run_metadata,
                )
            return PreparedTurn(
                request_id=active_request_id,
                conversation_id=active_conversation_id,
                decision=decision,
                route_result=route_result,
                agent_name=agent.name,
                model_id=model_id,
                messages=[],
                final_message=decision.decision_summary,
                final_status="error",
                warnings=warnings,
                project_id=project.id if project else None,
                actor=actor,
                history=memory_context.history,
                context_sections=context_sections,
                task_plan_id=task_plan.id,
                run_metadata=run_metadata,
            )

        tool_outputs, tool_output_truncated = self._bound_tool_outputs(tool_outputs)
        run_metadata["context_category_character_usage"]["tool_output"] = sum(
            len(item) for item in tool_outputs
        )
        run_metadata["context_category_truncated"]["tool_output"] = tool_output_truncated
        prompt_parts, prompt_citations = await self._prompt_parts(
            message=message,
            decision=decision,
            project=project,
            tool_outputs=tool_outputs,
            memory_context=memory_context,
        )
        citations.extend(prompt_citations)
        return PreparedTurn(
            request_id=active_request_id,
            conversation_id=active_conversation_id,
            decision=decision,
            route_result=route_result,
            agent_name=agent.name,
            model_id=model_id,
            messages=self._conversation_chat_messages(
                system_prompt=agent.system_prompt,
                memory_context=memory_context,
                current_prompt="\n\n".join(prompt_parts),
            ),
            citations=citations,
            pending_approval=pending_approval,
            warnings=warnings,
            project_id=project.id if project else None,
            actor=actor,
            history=memory_context.history,
            context_sections=context_sections,
            task_plan_id=task_plan.id,
            run_metadata=run_metadata,
        )
