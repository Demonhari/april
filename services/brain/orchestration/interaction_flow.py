# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from agents.schemas import AgentResult
from services.april_runtime.schemas import GenerationOptions
from services.brain.capabilities import (
    collect_runtime_self_evidence,
    is_conversation_recall_request,
    is_identity_request,
    is_self_introspection_request,
    is_voice_capability_request,
    render_conversation_recall_response,
    render_self_status,
    render_voice_capability_response,
    voice_capability_intent,
)
from services.brain.execution import PreparedTurn
from services.brain.feedback_classifier import classify_implicit_correction
from services.brain.intelligence_ladder import (
    ChatMode,
)
from services.brain.orchestration.models import StreamEventName
from services.brain.request_context import RequestContext
from services.brain.response_handling import ReasoningStreamFilter, sanitize_model_output
from services.evolution.feedback_eval import stage_feedback_eval_case
from skills.playbooks.runner import PlaybookRunResult


class InteractionFlow:
    async def _application_owned_response(
        self,
        message: str,
        *,
        conversation_id: str | None,
        request_id: str,
        actor: str,
        project_id: str | None,
        repo_path: str | None,
        agent_name: str,
        mode: ChatMode,
        request_context: RequestContext | None = None,
    ) -> AgentResult | None:
        if not (
            is_identity_request(message)
            or is_self_introspection_request(message)
            or is_voice_capability_request(message)
            or is_conversation_recall_request(message)
        ):
            return None
        active_request_context = request_context or RequestContext.unknown()
        recall_request = is_conversation_recall_request(message)
        voice_intent = voice_capability_intent(message)
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
        prepared_context = (
            await self.conversation_context.prepare(
                conversation_id=active_conversation_id,
                request_id=request_id,
            )
            if recall_request
            else None
        )
        await self.memory.add_message(active_conversation_id, "user", message)
        identity = is_identity_request(message)
        voice_capability = is_voice_capability_request(message)
        runtime_evidence = (
            await collect_runtime_self_evidence(self.runtime_client)
            if not identity and not voice_capability and not recall_request
            else None
        )
        final_message = (
            "I'm APRIL, your personal local assistant."
            if identity
            else render_conversation_recall_response(
                prepared_context.recent_messages if prepared_context is not None else [],
                history_complete=(
                    prepared_context.history_complete if prepared_context is not None else False
                ),
            )
            if recall_request
            else render_voice_capability_response(
                active_request_context,
                intent=voice_intent or "support",
            )
            if voice_capability
            else render_self_status(
                settings=self.settings,
                agent_registry=self.agent_registry,
                tool_registry=self.tool_registry,
                model_registry=getattr(self, "model_registry", None),
                runtime_evidence=runtime_evidence,
                request_context=active_request_context,
            )
        )
        await self.memory.add_message(active_conversation_id, "assistant", final_message)
        response_kind = (
            "identity"
            if identity
            else "conversation_recall"
            if recall_request
            else "voice_capability"
            if voice_capability
            else "self_status"
        )
        metadata = {
            "application_owned": True,
            "response_kind": response_kind,
            "request_id": request_id,
            "chat_mode": mode,
            "intelligence_rung": 0,
            "routing_method": "application_owned",
        }
        await self.memory.record_agent_run(
            conversation_id=active_conversation_id,
            agent=agent_name,
            status="ok",
            model_id=None,
            summary=f"application-owned {response_kind} response",
            metadata=metadata,
        )
        return AgentResult(
            status="ok",
            final_message=final_message,
            conversation_id=active_conversation_id,
            metadata=metadata,
        )

    async def _maybe_record_implicit_correction(
        self, message: str, conversation_id: str | None
    ) -> None:
        """Record a conservative implicit negative signal on clear corrections.

        Deterministic prefix classification only (no model), bound to the most
        recent agent run of an *existing* conversation. Failures are swallowed:
        feedback capture must never break the chat turn itself.
        """
        if conversation_id is None:
            return
        marker = classify_implicit_correction(message)
        if marker is None:
            return
        try:
            agent_run_id = await self.memory.latest_agent_run_id(conversation_id=conversation_id)
            if agent_run_id is None:
                return
            await self.routing_reliability.mark_negative_feedback(
                agent_run_id=agent_run_id,
                implicit_correction=True,
            )
            if self.overlay_manager is not None:
                from services.evolution.rollouts import RolloutService

                await RolloutService(
                    self.settings,
                    self.memory.database,
                    audit=self.approvals.audit,
                ).record_signal_for_agent_run(
                    agent_run_id=agent_run_id,
                    signal="user_correction",
                )
            record = await self.memory.record_feedback_event(
                rating="bad",
                reason=f"implicit_correction: {marker}",
                conversation_id=conversation_id,
                agent_run_id=agent_run_id,
            )
            self.approvals.audit.write(
                {
                    "event_type": "feedback_recorded",
                    "actor": "local-user",
                    "rating": record.rating,
                    "kind": "implicit_correction",
                    "reason_length": len(record.reason or ""),
                    "agent_run_bound": True,
                }
            )
            await stage_feedback_eval_case(
                self.settings,
                self.memory,
                record,
                kind="implicit_correction",
                audit=self.approvals.audit,
            )
        except Exception:
            return

    async def _maybe_run_playbook(
        self,
        message: str,
        *,
        conversation_id: str | None,
        request_id: str,
        actor: str,
        project_id: str | None,
    ) -> AgentResult | None:
        """Route an unambiguous active-playbook trigger straight to the runner.

        Ambiguous or absent trigger matches return ``None`` so the turn falls
        back to normal Brain routing. Playbook steps run through the standard
        tool execution path, so Level 3+ steps still raise exact-action
        approvals here exactly as they would anywhere else.
        """
        if self.playbook_loader is None or self.playbook_runner is None:
            return None
        try:
            playbook = self.playbook_loader.match_trigger(message)
        except Exception:
            return None
        if playbook is None:
            return None
        active_conversation_id = conversation_id or await self.memory.create_conversation(
            project_id=project_id,
            actor=actor,
        )
        if conversation_id is not None:
            await self.memory.ensure_conversation(
                active_conversation_id,
                project_id=project_id,
                actor=actor,
            )
        await self.memory.add_message(active_conversation_id, "user", message)
        run = await self.playbook_runner.run(
            playbook,
            conversation_id=active_conversation_id,
            project_id=project_id,
            actor=actor,
            source="api",
        )
        result = self._playbook_agent_result(playbook.id, run, active_conversation_id)
        if result.status != "pending_approval":
            await self.memory.add_message(active_conversation_id, "assistant", result.final_message)
        await self.memory.record_agent_run(
            conversation_id=active_conversation_id,
            agent="playbook_runner",
            status=result.status,
            model_id=None,
            summary=f"playbook {playbook.id} via trigger match",
            metadata={
                "playbook_id": playbook.id,
                "playbook_run_id": run.run_id,
                "routing_method": "playbook_trigger",
            },
        )
        return result

    @staticmethod
    def _playbook_agent_result(
        playbook_id: str, run: PlaybookRunResult, conversation_id: str
    ) -> AgentResult:
        if run.status == "pending_approval":
            approval = next(
                (step.approval for step in run.steps if step.approval is not None), None
            )
            return AgentResult(
                status="pending_approval",
                final_message=(
                    f"Playbook {playbook_id} paused after {run.steps_completed} step(s): "
                    "the next step requires exact-action approval."
                ),
                conversation_id=conversation_id,
                pending_approval=approval,
            )
        if run.status == "completed":
            return AgentResult(
                status="ok",
                final_message=(f"Playbook {playbook_id} completed {run.steps_completed} step(s)."),
                conversation_id=conversation_id,
            )
        return AgentResult(
            status="error",
            final_message=(
                f"Playbook {playbook_id} failed after {run.steps_completed} completed step(s)."
            ),
            conversation_id=conversation_id,
        )

    async def _run_standard_prepared(self, prepared: PreparedTurn, message: str) -> AgentResult:
        if prepared.structured_agent:
            return await self._run_structured_prepared(prepared, message)
        if prepared.pending_approval is not None:
            return await self._finish_pending(prepared)
        if prepared.final_message is not None:
            return await self._finish_message(prepared, prepared.final_message)

        response = await self.runtime_client.chat(
            model_id=prepared.model_id,
            messages=prepared.messages,
            options=GenerationOptions(enable_thinking=False),
            request_id=prepared.request_id,
        )
        final_message = sanitize_model_output(response.content)
        if not final_message:
            result = AgentResult(
                status="error",
                final_message="APRIL did not receive a complete user-facing answer.",
                conversation_id=prepared.conversation_id,
                local_citations=prepared.citations,
                warnings=[
                    *prepared.warnings,
                    *response.warnings,
                    "Model returned only control text.",
                ],
                usage=response.usage.model_dump(),
                metadata=dict(prepared.run_metadata),
            )
            agent_run_id = await self.memory.record_agent_run(
                conversation_id=prepared.conversation_id,
                agent=prepared.agent_name,
                status=result.status,
                model_id=prepared.model_id,
                summary=prepared.decision.decision_summary,
                metadata=prepared.run_metadata,
            )
            await self._record_routing_outcome(
                prepared, agent_run_id=agent_run_id, final_status=result.status
            )
            await self._update_task_status(prepared, "error")
            return result
        await self.memory.add_message(prepared.conversation_id, "assistant", final_message)
        result = AgentResult(
            status="ok",
            final_message=final_message,
            conversation_id=prepared.conversation_id,
            local_citations=prepared.citations,
            warnings=[*prepared.warnings, *response.warnings],
            usage=response.usage.model_dump(),
            metadata=dict(prepared.run_metadata),
        )
        agent_run_id = await self.memory.record_agent_run(
            conversation_id=prepared.conversation_id,
            agent=prepared.agent_name,
            status=result.status,
            model_id=prepared.model_id,
            summary=prepared.decision.decision_summary,
            metadata=prepared.run_metadata,
        )
        await self._record_routing_outcome(
            prepared,
            agent_run_id=agent_run_id,
            final_status=result.status,
        )
        await self._update_task_status(prepared, "completed")
        return result

    async def _run_verified_prepared(self, prepared: PreparedTurn, message: str) -> AgentResult:
        if prepared.structured_agent or prepared.pending_approval is not None:
            return await self._run_standard_prepared(prepared, message)
        if prepared.final_message is not None:
            return await self._run_standard_prepared(prepared, message)

        response = await self.runtime_client.chat(
            model_id=prepared.model_id,
            messages=prepared.messages,
            options=GenerationOptions(
                max_output_tokens=self.settings.deep_mode.verified_draft_tokens,
                enable_thinking=False,
            ),
            request_id=prepared.request_id,
        )
        draft = sanitize_model_output(response.content)
        verified = await self.intelligence_ladder.verify_and_revise(
            message=message,
            initial_answer=draft,
            model_id=prepared.model_id,
            request_id=prepared.request_id,
            trusted_context=prepared.trusted_context,
        )
        final_message = sanitize_model_output(verified.final_message)
        answer_available = bool(final_message)
        if not answer_available:
            final_message = "APRIL did not receive a complete user-facing answer."
            verified.status = "unavailable"
        if answer_available:
            await self.memory.add_message(prepared.conversation_id, "assistant", final_message)
        prepared.run_metadata.update(verified.metadata)
        result = AgentResult(
            status="ok" if answer_available else "error",
            final_message=final_message,
            conversation_id=prepared.conversation_id,
            local_citations=prepared.citations,
            warnings=[*prepared.warnings, *response.warnings, *verified.warnings],
            usage={**response.usage.model_dump(), **verified.usage},
            metadata=dict(prepared.run_metadata),
        )
        agent_run_id = await self.memory.record_agent_run(
            conversation_id=prepared.conversation_id,
            agent=prepared.agent_name,
            status=result.status,
            model_id=prepared.model_id,
            summary=prepared.decision.decision_summary,
            metadata=prepared.run_metadata,
        )
        await self._record_routing_outcome(
            prepared,
            agent_run_id=agent_run_id,
            final_status=result.status,
            regeneration_or_retry=True,
        )
        await self._update_task_status(prepared, "completed" if answer_available else "error")
        return result

    async def stream_chat(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        request_id: str | None = None,
        actor: str = "local-user",
        project_id: str | None = None,
        repo_path: str | None = None,
        mode: ChatMode = "standard",
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[tuple[StreamEventName, dict[str, Any]]]:
        active_request_id = request_id or str(uuid.uuid4())
        application_result = await self._application_owned_response(
            message,
            conversation_id=conversation_id,
            request_id=active_request_id,
            actor=actor,
            project_id=project_id,
            repo_path=repo_path,
            agent_name="general_agent",
            mode=mode,
            request_context=request_context,
        )
        if application_result is not None:
            yield (
                "meta",
                {
                    **application_result.metadata,
                    "conversation_id": application_result.conversation_id,
                    "agent": "general_agent",
                    "model_id": None,
                },
            )
            yield ("final_answer", {"message": application_result.final_message})
            yield ("token", {"text": application_result.final_message})
            yield ("usage", application_result.usage)
            yield ("done", {"finish_reason": "stop"})
            return
        reminder_reflex = await self._maybe_direct_reminder_reflex(
            message,
            conversation_id=conversation_id,
            request_id=active_request_id,
            actor=actor,
            project_id=project_id,
            repo_path=repo_path,
        )
        if reminder_reflex is not None:
            yield ("final_answer", {"message": reminder_reflex.final_message})
            yield ("token", {"text": reminder_reflex.final_message})
            yield ("usage", reminder_reflex.usage)
            yield ("done", {"finish_reason": "stop"})
            return
        playbook_result = await self._maybe_run_playbook(
            message,
            conversation_id=conversation_id,
            request_id=active_request_id,
            actor=actor,
            project_id=project_id,
        )
        if playbook_result is not None:
            if playbook_result.status == "pending_approval":
                yield (
                    "approval_required",
                    self._approval_required_payload(
                        approval=playbook_result.pending_approval or {},
                        message=playbook_result.final_message,
                        proposed_changes=[],
                    ),
                )
                yield ("done", {"finish_reason": "approval_required"})
                return
            yield ("final_answer", {"message": playbook_result.final_message})
            yield ("token", {"text": playbook_result.final_message})
            yield ("usage", playbook_result.usage)
            yield (
                "done",
                {"finish_reason": "stop" if playbook_result.status == "ok" else "error"},
            )
            return
        prepared = await self._prepare_turn(
            message,
            conversation_id=conversation_id,
            request_id=active_request_id,
            actor=actor,
            project_id=project_id,
            repo_path=repo_path,
            structured_specialists=True,
            mode=mode,
            request_context=request_context,
        )
        selection = self._select_intelligence_rung(prepared, message=message, mode=mode)
        self._schedule_agent_prewarm(prepared)
        yield (
            "meta",
            {
                "request_id": prepared.request_id,
                "conversation_id": prepared.conversation_id,
                "agent": prepared.agent_name,
                "model_id": prepared.model_id,
                "routing_method": prepared.route_result.route_source.value,
                "route_source": prepared.route_result.route_source.value,
                "citations": [citation.model_dump() for citation in prepared.citations],
                "run_metadata": prepared.run_metadata,
                "chat_mode": selection.mode,
                "intelligence_rung": selection.rung,
            },
        )
        yield (
            "routing",
            {
                "intent": prepared.decision.intent,
                "agent": prepared.agent_name,
                "model_id": prepared.model_id,
                "routing_method": prepared.route_result.route_source.value,
                "route_source": prepared.route_result.route_source.value,
                "decision_summary": prepared.decision.decision_summary,
                "run_metadata": prepared.run_metadata,
                "confidence": prepared.decision.confidence,
                "chat_mode": selection.mode,
                "intelligence_rung": selection.rung,
            },
        )
        ladder_result = await self._maybe_run_ladder(prepared, message, selection)
        if ladder_result is not None:
            yield ("final_answer", {"message": ladder_result.final_message})
            yield ("token", {"text": ladder_result.final_message})
            yield ("usage", ladder_result.usage)
            yield ("done", {"finish_reason": "stop" if ladder_result.status == "ok" else "error"})
            return
        if selection.rung == 2:
            result = await self._run_verified_prepared(prepared, message)
            yield ("final_answer", {"message": result.final_message})
            yield ("token", {"text": result.final_message})
            yield ("usage", result.usage)
            yield ("done", {"finish_reason": "stop" if result.status == "ok" else "error"})
            return
        if prepared.structured_agent:
            yield (
                "agent_iteration",
                {
                    "agent": prepared.agent_name,
                    "status": "started",
                    "structured": True,
                },
            )
            result = await self._run_structured_prepared(prepared, message)
            for request in result.tool_requests:
                yield ("tool_request", request)
            if result.pending_approval is not None:
                yield (
                    "approval_required",
                    self._approval_required_payload(
                        approval=result.pending_approval,
                        message=result.final_message,
                        proposed_changes=result.proposed_changes,
                    ),
                )
                yield ("done", {"finish_reason": "approval_required"})
                return
            if result.status == "ok":
                yield ("final_answer", {"message": result.final_message})
                yield ("token", {"text": result.final_message})
                yield ("usage", result.usage)
                yield ("done", {"finish_reason": "stop"})
                return
            yield (
                "error",
                {
                    "message": result.final_message,
                    "status": result.status,
                    "warnings": result.warnings,
                },
            )
            yield ("done", {"finish_reason": "error"})
            return
        if prepared.pending_approval is not None:
            yield (
                "approval_required",
                self._approval_required_payload(
                    approval=prepared.pending_approval,
                    message=prepared.final_message,
                    proposed_changes=prepared.proposed_changes,
                ),
            )
            await self._finish_pending(prepared)
            yield ("done", {"finish_reason": "approval_required"})
            return
        if prepared.final_message is not None:
            if prepared.final_status == "ok":
                yield ("final_answer", {"message": prepared.final_message})
                yield ("token", {"text": prepared.final_message})
                yield ("usage", {})
                await self._finish_message(prepared, prepared.final_message)
                yield ("done", {"finish_reason": "stop"})
                return
            yield ("error", {"message": prepared.final_message, "warnings": prepared.warnings})
            await self._finish_message(prepared, prepared.final_message)
            yield ("done", {"finish_reason": "error"})
            return

        chunks: list[str] = []
        output_filter = ReasoningStreamFilter()
        finish_reason = "stop"
        try:
            async for raw_event in self.runtime_client.stream(
                model_id=prepared.model_id,
                messages=prepared.messages,
                options=GenerationOptions(enable_thinking=False),
                request_id=prepared.request_id,
            ):
                event_name, payload = self._parse_runtime_stream_event(raw_event)
                if event_name == "token":
                    text = output_filter.feed(str(payload.get("text", "")))
                    if text:
                        chunks.append(text)
                        yield ("token", {"text": text})
                elif event_name == "usage":
                    yield ("usage", payload)
                elif event_name == "error":
                    yield ("error", payload)
                    finish_reason = "error"
                    break
                elif event_name == "done":
                    finish_reason = str(payload.get("finish_reason", "stop"))
                    break
                elif event_name == "meta":
                    continue
        except Exception as exc:
            yield ("error", {"message": str(exc)})
            finish_reason = "error"

        tail = output_filter.finish()
        if tail:
            chunks.append(tail)
            yield ("token", {"text": tail})
        content = sanitize_model_output("".join(chunks))
        successful = finish_reason not in {"error", "cancelled"} and bool(content)
        if successful:
            await self.memory.add_message(prepared.conversation_id, "assistant", content)
        agent_run_id = await self.memory.record_agent_run(
            conversation_id=prepared.conversation_id,
            agent=prepared.agent_name,
            status="ok" if successful else "error",
            model_id=prepared.model_id,
            summary=prepared.decision.decision_summary,
            metadata=prepared.run_metadata,
        )
        await self._record_routing_outcome(
            prepared,
            agent_run_id=agent_run_id,
            final_status="ok" if successful else "error",
        )
        await self._update_task_status(
            prepared,
            "completed" if successful else "error",
        )
        if not successful and finish_reason not in {"error", "cancelled"}:
            yield ("error", {"message": "APRIL did not receive a complete user-facing answer."})
        yield ("done", {"finish_reason": finish_reason})
