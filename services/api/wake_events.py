from __future__ import annotations

import contextlib
from typing import Any

from services.api.dependencies import ApiContainer
from services.evolution.feedback_eval import stage_feedback_eval_case
from services.wake.feedback import WakeFeedback, classify_wake_feedback
from services.wake.schemas import WakeEvent


async def _handle_wake_event(
    active: ApiContainer, event: WakeEvent, *, request_id: str
) -> dict[str, Any]:
    """Converge one wake event: resolve the session, then route optional text.

    ``event.text`` (typed or STT-confirmed command) is executed as a normal chat
    turn inside the session's conversation — through the standard orchestrator,
    Brain routing, and permission engine. Wake events never bypass anything.
    """
    session_manager = active.require_session_manager()
    resolution = await session_manager.handle_wake(event)
    active.approvals.audit.write(
        {
            "event_type": "wake_event",
            "request_id": request_id,
            "actor": "local-user",
            "source": event.source,
            "session_id": resolution.session_id,
            "joined_existing": resolution.joined_existing,
            "score": event.score,
            "reason": event.reason,
            "transcript_present": bool(event.text),
        }
    )
    payload: dict[str, Any] = {
        "request_id": request_id,
        **resolution.model_dump(),
    }
    if event.text:
        feedback = classify_wake_feedback(event.text)
        if feedback is not None:
            payload["result"] = await _record_wake_feedback(
                active,
                feedback,
                session_id=resolution.session_id,
                conversation_id=resolution.conversation_id,
                request_id=request_id,
            )
            await session_manager.touch(resolution.session_id)
            return payload
        async with session_manager.interaction(resolution.conversation_id):
            result = await active.orchestrator.chat(
                event.text,
                conversation_id=resolution.conversation_id,
                request_id=request_id,
            )
        payload["result"] = result.model_dump()
    return payload


async def _record_wake_feedback(
    active: ApiContainer,
    feedback: WakeFeedback,
    *,
    session_id: str,
    conversation_id: str | None,
    request_id: str,
) -> dict[str, Any]:
    agent_run_id = await active.memory.latest_agent_run_id(conversation_id=conversation_id)
    if agent_run_id is None:
        active.approvals.audit.write(
            {
                "event_type": "wake_feedback_noop",
                "request_id": request_id,
                "actor": "local-user",
                "kind": "wake_feedback",
                "rating": feedback.rating,
                "reason_length": len(feedback.phrase),
                "agent_run_bound": False,
            }
        )
        return {
            "final_message": "I do not have a recent answer to attach that feedback to.",
            "feedback_recorded": False,
        }
    record = await active.memory.record_feedback_event(
        rating=feedback.rating,
        reason=f"wake_feedback: {feedback.phrase}",
        session_id=session_id,
        conversation_id=conversation_id,
        agent_run_id=agent_run_id,
    )
    active.approvals.audit.write(
        {
            "event_type": "feedback_recorded",
            "request_id": request_id,
            "actor": "local-user",
            "kind": "wake_feedback",
            "rating": record.rating,
            "reason_length": len(record.reason or ""),
            "agent_run_bound": True,
        }
    )
    if record.rating == "bad":
        with contextlib.suppress(Exception):
            await stage_feedback_eval_case(
                active.settings,
                active.memory,
                record,
                kind="wake_feedback",
                audit=active.approvals.audit,
            )
    return {"final_message": "Feedback recorded.", "feedback_recorded": True}
