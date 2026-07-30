# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import re
from typing import Any, Literal

from agents.schemas import AgentResult, ProposedChange
from april_common.time import parse_utc_iso
from services.brain.execution import PreparedTurn
from services.brain.intelligence_ladder import (
    ChatMode,
    LadderRun,
    LadderSelection,
)
from services.memory.policy import MemoryPolicy
from services.memory.schemas import ReminderRecord


class RoutingFlow:
    @staticmethod
    def _approval_required_payload(
        *,
        approval: dict[str, Any],
        message: str | None,
        proposed_changes: list[ProposedChange],
    ) -> dict[str, Any]:
        return {
            "approval": approval,
            "message": message,
            "proposed_changes": [change.model_dump() for change in proposed_changes],
        }

    def _select_intelligence_rung(
        self,
        prepared: PreparedTurn,
        *,
        message: str,
        mode: ChatMode,
    ) -> LadderSelection:
        selection = self.intelligence_ladder.select(
            message=message,
            decision=prepared.decision,
            mode=mode,
            effective_confidence=prepared.route_result.effective_confidence,
        )
        prepared.run_metadata.update(
            {
                "chat_mode": selection.mode,
                "intelligence_rung": selection.rung,
                "intelligence_reason": selection.reason,
                "routing_confidence": prepared.route_result.effective_confidence,
                "raw_routing_confidence": prepared.route_result.raw_model_confidence,
                "historical_routing_reliability": (prepared.route_result.historical_reliability),
                "effective_routing_confidence": prepared.route_result.effective_confidence,
                "routing_reliability_sample_count": (
                    prepared.route_result.reliability_sample_count
                ),
                "routing_confidence_source": prepared.route_result.confidence_source,
                "route_source": prepared.route_result.route_source.value,
                "matched_rule": prepared.route_result.matched_rule,
                "high_stakes": selection.high_stakes,
            }
        )
        return selection

    def _schedule_agent_prewarm(self, prepared: PreparedTurn) -> None:
        if self.agent_pool is None:
            return
        try:
            self.agent_pool.schedule_prewarm(
                agent=prepared.agent_name,
                model_id=prepared.model_id,
                request_id=prepared.request_id,
            )
        except Exception:
            return

    async def _maybe_run_ladder(
        self,
        prepared: PreparedTurn,
        message: str,
        selection: LadderSelection,
    ) -> AgentResult | None:
        if selection.rung != 0 and selection.mode == "standard" and not selection.high_stakes:
            reflex_result = await self._maybe_memory_reflex(prepared, message)
            if reflex_result is not None:
                return reflex_result
        if selection.rung == 0:
            reminder_reflex = await self._maybe_reminder_reflex(prepared, message)
            if reminder_reflex is not None:
                return reminder_reflex
            run = LadderRun(
                status="ok",
                final_message=self.intelligence_ladder.reflex_answer(message),
                mode=selection.mode,
                rung=0,
                model_id=prepared.model_id,
                metadata={
                    "mode": selection.mode,
                    "intelligence_rung": 0,
                    "deterministic": True,
                },
            )
            return await self._finish_ladder_run(prepared, run)
        if selection.rung == 3:
            run = await self.intelligence_ladder.run_deep(
                message=message,
                prompt_messages=prepared.messages,
                fallback_model_id=prepared.model_id,
                request_id=prepared.request_id,
            )
            return await self._finish_ladder_run(prepared, run)
        if selection.rung == 4:
            run = await self.intelligence_ladder.run_council(
                message=message,
                prompt_messages=prepared.messages,
                fallback_model_id=prepared.model_id,
                request_id=prepared.request_id,
            )
            return await self._finish_ladder_run(prepared, run)
        return None

    async def _maybe_direct_reminder_reflex(
        self,
        message: str,
        *,
        conversation_id: str | None,
        request_id: str,
        actor: str,
        project_id: str | None,
        repo_path: str | None,
    ) -> AgentResult | None:
        kind = self.intelligence_ladder.reminder_reflex_kind(message)
        if kind is None:
            return None
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
        await self.memory.add_message(active_conversation_id, "user", message)
        reminders = await self._reminders_for_reflex(kind)
        final_message = self.intelligence_ladder.reminder_reflex_answer(reminders, kind=kind)
        await self.memory.add_message(active_conversation_id, "assistant", final_message)
        metadata: dict[str, Any] = {
            "chat_mode": "standard",
            "intelligence_rung": 0,
            "intelligence_reason": "exact reminder reflex",
            "routing_confidence": 1.0,
            "high_stakes": False,
            "mode": "standard",
            "deterministic": True,
            "reflex": f"reminders_{kind}",
            "routing_method": "deterministic_reflex",
            "permission_level": 0,
            "request_id": request_id,
        }
        await self.memory.record_conversation_event(
            conversation_id=active_conversation_id,
            event_type="reflex_decision",
            payload=metadata,
        )
        await self.memory.record_agent_run(
            conversation_id=active_conversation_id,
            agent="general_agent",
            status="ok",
            model_id=None,
            summary="Local reminder reflex",
            metadata=metadata,
        )
        return AgentResult(
            status="ok",
            final_message=final_message,
            conversation_id=active_conversation_id,
            metadata=metadata,
        )

    async def _maybe_reminder_reflex(
        self, prepared: PreparedTurn, message: str
    ) -> AgentResult | None:
        kind = self.intelligence_ladder.reminder_reflex_intent(message, prepared.decision)
        if kind is None:
            return None
        reminders = await self._reminders_for_reflex(kind)
        run = LadderRun(
            status="ok",
            final_message=self.intelligence_ladder.reminder_reflex_answer(reminders, kind=kind),
            mode="standard",
            rung=0,
            model_id=prepared.model_id,
            metadata={
                "mode": "standard",
                "intelligence_rung": 0,
                "deterministic": True,
                "reflex": f"reminders_{kind}",
            },
        )
        return await self._finish_ladder_run(prepared, run)

    async def _reminders_for_reflex(self, kind: str) -> list[ReminderRecord]:
        reminders = await self.memory.list_reminders()
        if kind != "today":
            return reminders
        now = self.intelligence_ladder.clock()
        today = now.date()
        local_tz = now.tzinfo
        today_reminders = []
        for reminder in reminders:
            if reminder.due_at is None or reminder.fired_at is not None:
                continue
            try:
                due = parse_utc_iso(reminder.due_at)
            except (TypeError, ValueError):
                continue
            due_local = due.astimezone(local_tz) if local_tz is not None else due
            if due_local.date() == today:
                today_reminders.append(reminder)
        return today_reminders

    async def _maybe_memory_reflex(
        self, prepared: PreparedTurn, message: str
    ) -> AgentResult | None:
        """R0 reflex for a unique, token-exact durable-memory recall hit.

        The reflex answers with stored memory verbatim (labelled as such) and
        never calls a model. It requires deterministic recall phrasing, a
        read-only decision, and exactly one active non-sensitive memory whose
        content contains every subject token — ambiguity or misses always fall
        back to normal routing.
        """
        subject = self.intelligence_ladder.memory_recall_subject(message, prepared.decision)
        if subject is None:
            return None
        try:
            candidates = await self.memory.search_memories(subject)
        except Exception:
            return None
        policy = MemoryPolicy()
        subject_tokens = set(re.findall(r"[a-z0-9_]+", subject.lower()))
        subject_tokens -= {"the", "a", "an", "of", "is", "was"}
        if not subject_tokens:
            return None
        matches = []
        for candidate in candidates:
            if policy.is_sensitive(candidate.content):
                continue
            content_tokens = set(re.findall(r"[a-z0-9_]+", candidate.content.lower()))
            if subject_tokens.issubset(content_tokens):
                matches.append(candidate)
        if len(matches) != 1:
            return None
        hit = matches[0]
        run = LadderRun(
            status="ok",
            final_message=self.intelligence_ladder.memory_reflex_answer(hit.content),
            mode="standard",
            rung=0,
            model_id=prepared.model_id,
            metadata={
                "mode": "standard",
                "intelligence_rung": 0,
                "deterministic": True,
                "reflex": "memory_hit",
                "memory_id": hit.id,
            },
        )
        return await self._finish_ladder_run(prepared, run)

    async def _finish_ladder_run(self, prepared: PreparedTurn, run: LadderRun) -> AgentResult:
        prepared.run_metadata.update(run.metadata)
        if run.candidates:
            prepared.run_metadata["council_candidates"] = [
                {
                    "responder_id": candidate.responder_id,
                    "score": candidate.score,
                    "rationale": candidate.rationale,
                }
                for candidate in run.candidates
            ]
        if run.status == "ok":
            await self.memory.add_message(prepared.conversation_id, "assistant", run.final_message)
        status: Literal["ok", "unavailable"] = "ok" if run.status == "ok" else "unavailable"
        result = AgentResult(
            status=status,
            final_message=run.final_message,
            conversation_id=prepared.conversation_id,
            local_citations=prepared.citations,
            warnings=[*prepared.warnings, *run.warnings],
            usage=run.usage,
            metadata=dict(prepared.run_metadata),
        )
        agent_run_id = await self.memory.record_agent_run(
            conversation_id=prepared.conversation_id,
            agent=prepared.agent_name,
            status=result.status,
            model_id=run.model_id or prepared.model_id,
            summary=prepared.decision.decision_summary,
            metadata=prepared.run_metadata,
        )
        await self._record_routing_outcome(
            prepared,
            agent_run_id=agent_run_id,
            final_status=result.status,
        )
        await self._update_task_status(
            prepared,
            "completed" if result.status == "ok" else "error",
        )
        return result
