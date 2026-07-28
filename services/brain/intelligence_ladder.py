from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from agents.registry import AgentRegistry
from april_common.errors import AprilError
from april_common.settings import AprilSettings
from services.april_runtime.client import RuntimeClient
from services.april_runtime.schemas import ChatMessage, GenerationOptions, ResponseFormat
from services.brain.reasoning_resolver import resolve_reasoning_model
from services.brain.schemas import BrainDecision
from services.evolution.versions import active_ladder_thresholds
from services.memory.schemas import ReminderRecord

ChatMode = Literal["standard", "deep", "council"]
LadderStatus = Literal["ok", "unavailable"]
ReminderReflexKind = Literal["all", "today"]

_MODE_ANNOUNCEMENTS: dict[int, str] = {
    0: "Mode: reflex (local deterministic answer).",
    2: "Mode: verified (local self-check).",
    3: "Mode: deep (local reasoning).",
    4: "Mode: council (local best-of-N).",
}

_DEEP_PHRASES = (
    "/deep",
    "think hard",
    "think harder",
    "think deeply",
    "think carefully",
    "reason step by step",
)
# Recall-question shapes eligible for the R0 memory reflex. Deliberately
# narrow: only first-person possessive recall ("my X") qualifies.
_MEMORY_RECALL_PATTERNS = (
    re.compile(r"^(?:what is|what's|whats) my (?P<subject>.+)$"),
    re.compile(r"^(?:do you remember|remind me(?: of| about)?) my (?P<subject>.+)$"),
)

_HIGH_STAKES_PHRASES = (
    "high stakes",
    "high-stakes",
    "big decision",
    "major decision",
    "important decision",
    "irreversible",
    "life changing",
    "life-changing",
)
_HIGH_STAKES_CONTEXT_PATTERNS = (
    re.compile(
        r"\b(?:transfer|send|pay|invest|borrow|loan|mortgage)\b.{0,60}\b(?:money|cash|funds?|dollars?|rupees?)\b"
    ),
    re.compile(
        r"\b(?:delete|drop|destroy|wipe|purge)\b.{0,60}\b(?:database|production|backup|account|keys?)\b"
    ),
    re.compile(
        r"\b(?:rotate|revoke|publish|expose|share)\b.{0,60}\b(?:credentials?|tokens?|secrets?|keys?)\b"
    ),
    re.compile(
        r"\b(?:security|privacy)\b.{0,60}\b"
        r"(?:incident|breach|vulnerability|credentials?|personal data)\b"
    ),
    re.compile(
        r"\b(?:irreversible|one-way|cannot be undone)\b.{0,80}\b"
        r"(?:migration|architecture|change|decision)\b"
    ),
)
_REMINDER_REFLEX_ALL_PHRASES = {"list my reminders", "what reminders do i have"}
_REMINDER_REFLEX_TODAY_PHRASES = {"any reminders today"}


@dataclass(frozen=True, slots=True)
class LadderSelection:
    mode: ChatMode
    rung: int
    reason: str
    announcement: str | None = None
    high_stakes: bool = False


@dataclass(frozen=True, slots=True)
class CouncilCandidate:
    responder_id: str
    content: str
    score: float = 0.0
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class _CouncilMember:
    role: str
    agent_name: str
    system_prompt: str
    model_id: str


@dataclass(slots=True)
class LadderRun:
    status: LadderStatus
    final_message: str
    mode: ChatMode
    rung: int
    model_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    candidates: list[CouncilCandidate] = field(default_factory=list)


class IntelligenceLadder:
    """Local-only chat mode selector and bounded reasoning executor."""

    def __init__(
        self,
        *,
        settings: AprilSettings,
        runtime_client: RuntimeClient,
        agent_registry: AgentRegistry,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.runtime_client = runtime_client
        self.agent_registry = agent_registry
        self.clock = clock or (lambda: datetime.now().astimezone())

    def select(
        self,
        *,
        message: str,
        decision: BrainDecision,
        mode: ChatMode,
        effective_confidence: float | None = None,
    ) -> LadderSelection:
        # The model may recommend escalation, but can never lower the bounded
        # deterministic classification.
        high_stakes = decision.high_stakes or self.is_high_stakes(message)
        if self.is_reflex_query(message, decision):
            return LadderSelection(
                mode=mode,
                rung=0,
                reason="trivial local lookup",
                announcement=_MODE_ANNOUNCEMENTS[0],
            )
        standard_fallback = LadderSelection(
            mode=mode,
            rung=1,
            reason="tool or approval path uses the standard permission flow",
            high_stakes=high_stakes,
        )
        if mode == "council":
            if self._can_use_reasoning_mode(decision):
                return LadderSelection(
                    mode=mode,
                    rung=4,
                    reason="explicit council mode",
                    announcement=_MODE_ANNOUNCEMENTS[4],
                    high_stakes=high_stakes,
                )
            return standard_fallback
        if mode == "deep":
            if self._can_use_reasoning_mode(decision):
                return LadderSelection(
                    mode=mode,
                    rung=3,
                    reason="explicit deep mode",
                    announcement=_MODE_ANNOUNCEMENTS[3],
                    high_stakes=high_stakes,
                )
            return standard_fallback
        if self._needs_verified_revision(message, decision):
            return LadderSelection(
                mode=mode,
                rung=2,
                reason="explicit verification request",
                announcement=_MODE_ANNOUNCEMENTS[2],
                high_stakes=high_stakes,
            )
        if not self._can_use_reasoning_mode(decision):
            return LadderSelection(
                mode=mode, rung=1, reason="standard route", high_stakes=high_stakes
            )
        # Safe, read-only requests escalate by stakes, phrases, and confidence.
        if high_stakes:
            return LadderSelection(
                mode=mode,
                rung=4,
                reason="high-stakes decision escalated to council",
                announcement=_MODE_ANNOUNCEMENTS[4],
                high_stakes=True,
            )
        normalized = _normalize(message)
        if any(phrase in normalized for phrase in _DEEP_PHRASES):
            return LadderSelection(
                mode=mode,
                rung=3,
                reason="deep-thinking phrase requested more reasoning",
                announcement=_MODE_ANNOUNCEMENTS[3],
            )
        thresholds = active_ladder_thresholds(self.settings)
        deep_threshold = thresholds["deep_confidence_threshold"]
        verified_threshold = thresholds["verified_confidence_threshold"]
        confidence = (
            decision.confidence
            if effective_confidence is None
            else min(1.0, max(0.0, effective_confidence))
        )
        if confidence < deep_threshold:
            return LadderSelection(
                mode=mode,
                rung=3,
                reason=(f"effective routing confidence {confidence:.2f} below {deep_threshold}"),
                announcement=_MODE_ANNOUNCEMENTS[3],
            )
        if confidence < verified_threshold:
            return LadderSelection(
                mode=mode,
                rung=2,
                reason=(
                    f"effective routing confidence {confidence:.2f} below {verified_threshold}"
                ),
                announcement=_MODE_ANNOUNCEMENTS[2],
            )
        return LadderSelection(mode=mode, rung=1, reason="standard route")

    def is_high_stakes(self, message: str) -> bool:
        """Bounded deterministic stakes tagging with contextual action pairs."""
        normalized = _normalize(message)
        return any(phrase in normalized for phrase in _HIGH_STAKES_PHRASES) or any(
            pattern.search(normalized) is not None for pattern in _HIGH_STAKES_CONTEXT_PATTERNS
        )

    def is_reflex_query(self, message: str, decision: BrainDecision) -> bool:
        if not self._can_use_reflex(decision):
            return False
        normalized = _normalize(message)
        return (
            normalized
            in {
                "what time is it",
                "what is the time",
                "current time",
                "time",
                "what day is it",
                "what is today",
                "today date",
                "today's date",
                "current date",
                "date",
            }
            or self.reminder_reflex_kind(message) is not None
        )

    def reminder_reflex_kind(self, message: str) -> ReminderReflexKind | None:
        normalized = _normalize(message)
        if normalized in _REMINDER_REFLEX_ALL_PHRASES:
            return "all"
        if normalized in _REMINDER_REFLEX_TODAY_PHRASES:
            return "today"
        return None

    def reminder_reflex_intent(
        self, message: str, decision: BrainDecision
    ) -> ReminderReflexKind | None:
        if not self._can_use_reflex(decision):
            return None
        return self.reminder_reflex_kind(message)

    def memory_recall_subject(self, message: str, decision: BrainDecision) -> str | None:
        """Deterministic recall-question detection for the R0 memory reflex.

        Returns the recall subject ("favorite editor" from "what is my favorite
        editor?") only for plain read-only turns. Retrieval scores are not
        calibrated confidence, so the reflex itself later requires a *unique*
        token-exact durable-memory match before answering without a model.
        """
        if decision.permission_level > 1 or decision.needs_confirmation:
            return None
        if decision.tools_needed or decision.planned_tool_calls:
            return None
        normalized = _normalize(message)
        for pattern in _MEMORY_RECALL_PATTERNS:
            match = pattern.match(normalized)
            if match:
                subject = match.group("subject").strip()
                if len(subject.split()) >= 1 and len(subject) >= 3:
                    return subject
        return None

    def memory_reflex_answer(self, memory_content: str) -> str:
        return f"{_MODE_ANNOUNCEMENTS[0]}\n\nFrom local memory: {memory_content}"

    def reminder_reflex_answer(
        self, reminders: list[ReminderRecord], *, kind: ReminderReflexKind
    ) -> str:
        if not reminders:
            scope = "today" if kind == "today" else "in local storage"
            return f"{_MODE_ANNOUNCEMENTS[0]}\n\nNo reminders found {scope}."
        heading = "Reminders today:" if kind == "today" else "Reminders:"
        lines = [heading]
        for reminder in reminders:
            due = f" (due {reminder.due_at})" if reminder.due_at else ""
            lines.append(f"- {reminder.content}{due}")
        return f"{_MODE_ANNOUNCEMENTS[0]}\n\n" + "\n".join(lines)

    def reflex_answer(self, message: str) -> str:
        now = self.clock()
        normalized = _normalize(message)
        if "time" in normalized:
            answer = now.strftime("It is %H:%M %Z on %Y-%m-%d.")
        else:
            answer = now.strftime("Today is %A, %Y-%m-%d.")
        return f"{_MODE_ANNOUNCEMENTS[0]}\n\n{answer}"

    async def run_deep(
        self,
        *,
        message: str,
        prompt_messages: list[ChatMessage],
        fallback_model_id: str,
        request_id: str,
    ) -> LadderRun:
        response = None
        model_id = fallback_model_id
        metadata = {
            "mode": "deep",
            "intelligence_rung": 3,
            "budget_seconds": self.settings.deep_mode.max_seconds,
            "model_resolution": _budget_unresolved_model_metadata(fallback_model_id),
        }
        try:
            async with asyncio.timeout(self.settings.deep_mode.max_seconds):
                resolution = await resolve_reasoning_model(
                    runtime_client=self.runtime_client,
                    fallback_model_id=fallback_model_id,
                )
                model_id = resolution.model_id
                metadata["model_resolution"] = resolution.metadata()
                user_prompt = self._reasoning_user_prompt(
                    mode="deep",
                    message=message,
                    prompt_messages=prompt_messages,
                )
                response = await self._bounded_chat(
                    model_id=model_id,
                    messages=[
                        ChatMessage(role="system", content=self._reasoning_system_prompt()),
                        ChatMessage(role="user", content=user_prompt),
                    ],
                    request_id=request_id,
                    max_output_tokens=self.settings.deep_mode.deep_tokens,
                )
        except (TimeoutError, AprilError, OSError):
            response = None
        if response is None:
            return LadderRun(
                status="unavailable",
                final_message=(
                    "Deep mode stopped at the configured local budget before a complete "
                    "answer was produced."
                ),
                mode="deep",
                rung=3,
                model_id=model_id,
                warnings=["Deep mode exceeded its configured local budget."],
                metadata=metadata,
            )
        return LadderRun(
            status="ok",
            final_message=f"{_MODE_ANNOUNCEMENTS[3]}\n\n{response.content}",
            mode="deep",
            rung=3,
            model_id=model_id,
            usage=response.usage.model_dump(),
            warnings=response.warnings,
            metadata=metadata,
        )

    async def verify_and_revise(
        self,
        *,
        message: str,
        initial_answer: str,
        model_id: str,
        request_id: str,
    ) -> LadderRun:
        metadata = {
            "mode": "standard",
            "intelligence_rung": 2,
            "budget_seconds": self.settings.deep_mode.max_seconds,
            "draft_token_budget": self.settings.deep_mode.verified_draft_tokens,
            "critique_token_budget": self.settings.deep_mode.verified_critique_tokens,
            "revision_token_budget": self.settings.deep_mode.verified_revision_tokens,
        }
        try:
            async with asyncio.timeout(self.settings.deep_mode.max_seconds):
                critique = await self._bounded_chat(
                    model_id=model_id,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=(
                                "You are APRIL's local verifier. Return exactly one JSON "
                                "object. Do not include hidden reasoning."
                            ),
                        ),
                        ChatMessage(
                            role="user",
                            content=(
                                "Check the answer for correctness, unsupported claims, and "
                                "missing caveats. Return "
                                '{"needs_revision": boolean, "critique": string}.\n'
                                f"User request:\n{message}\n\nAssistant answer:\n{initial_answer}"
                            ),
                        ),
                    ],
                    request_id=f"{request_id}-critique",
                    response_format=ResponseFormat(
                        type="json_object",
                        json_schema={
                            "type": "object",
                            "properties": {
                                "needs_revision": {"type": "boolean"},
                                "critique": {"type": "string"},
                            },
                            "required": ["needs_revision", "critique"],
                        },
                    ),
                    max_output_tokens=self.settings.deep_mode.verified_critique_tokens,
                )
                if critique is None:
                    raise TimeoutError
                payload = json.loads(critique.content)
                needs_revision = payload.get("needs_revision") is True
                critique_text = str(payload.get("critique") or "")[:2000]
                if not needs_revision:
                    return LadderRun(
                        status="ok",
                        final_message=f"{_MODE_ANNOUNCEMENTS[2]}\n\n{initial_answer}",
                        mode="standard",
                        rung=2,
                        model_id=model_id,
                        usage=critique.usage.model_dump(),
                        warnings=critique.warnings,
                        metadata={**metadata, "verification_reason": critique_text},
                    )
                revision = await self._bounded_chat(
                    model_id=model_id,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=(
                                "Revise the answer using the bounded critique. Return only "
                                "the final user-facing answer; do not expose hidden reasoning."
                            ),
                        ),
                        ChatMessage(
                            role="user",
                            content=(
                                f"User request:\n{message}\n\nDraft:\n{initial_answer}\n\n"
                                f"Critique:\n{critique_text}"
                            ),
                        ),
                    ],
                    request_id=f"{request_id}-revision",
                    max_output_tokens=self.settings.deep_mode.verified_revision_tokens,
                )
                if revision is None:
                    raise TimeoutError
        except (TimeoutError, AprilError, OSError, json.JSONDecodeError, TypeError):
            return LadderRun(
                status="ok",
                final_message=initial_answer,
                mode="standard",
                rung=2,
                model_id=model_id,
                warnings=["Verification was unavailable or invalid; kept the original answer."],
                metadata=metadata,
            )
        final_answer = revision.content.strip() or initial_answer
        usage = {
            key: int(critique.usage.model_dump().get(key, 0))
            + int(revision.usage.model_dump().get(key, 0))
            for key in {"input_tokens", "output_tokens", "total_tokens"}
        }
        return LadderRun(
            status="ok",
            final_message=f"{_MODE_ANNOUNCEMENTS[2]}\n\n{final_answer}",
            mode="standard",
            rung=2,
            model_id=model_id,
            usage=usage,
            warnings=[*critique.warnings, *revision.warnings],
            metadata={**metadata, "verification_reason": critique_text},
        )

    async def run_council(
        self,
        *,
        message: str,
        prompt_messages: list[ChatMessage],
        fallback_model_id: str,
        request_id: str,
    ) -> LadderRun:
        reasoning_fallback_model_id = self._agent_model_id("reasoning_agent", fallback_model_id)
        model_id = reasoning_fallback_model_id
        requested_council_mode = self.settings.deep_mode.council_mode
        members: list[_CouncilMember] = []
        raw_candidates: list[CouncilCandidate] = []
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        warnings: list[str] = []
        judged: list[CouncilCandidate] | None = None
        metadata: dict[str, Any] = {
            "mode": "council",
            "intelligence_rung": 4,
            "budget_seconds": self.settings.deep_mode.max_seconds,
            "model_resolution": _budget_unresolved_model_metadata(reasoning_fallback_model_id),
            "candidate_count": len(raw_candidates),
            "council_mode_requested": requested_council_mode,
            "council_mode_effective": requested_council_mode,
            "council_members": [],
        }
        try:
            async with asyncio.timeout(self.settings.deep_mode.max_seconds):
                resolution = await resolve_reasoning_model(
                    runtime_client=self.runtime_client,
                    fallback_model_id=reasoning_fallback_model_id,
                )
                model_id = resolution.model_id
                metadata["model_resolution"] = resolution.metadata()
                if requested_council_mode == "multi_agent":
                    proposed_members = self._multi_agent_council_members(
                        reasoning_model_id=model_id,
                        fallback_model_id=fallback_model_id,
                    )
                    distinct_models = {member.model_id for member in proposed_members}
                    if len(distinct_models) >= 2:
                        members = proposed_members
                    else:
                        metadata["council_mode_effective"] = "reasoning_n"
                        metadata["council_fallback_reason"] = (
                            "fewer_than_two_distinct_models_resolved"
                        )
                        members = self._reasoning_n_council_members(model_id)[
                            : self.settings.deep_mode.council_n
                        ]
                else:
                    members = self._reasoning_n_council_members(model_id)[
                        : self.settings.deep_mode.council_n
                    ]
                metadata["council_members"] = [member.role for member in members]
                metadata["council_member_models"] = {
                    member.role: member.model_id for member in members
                }
                metadata["council_member_agents"] = {
                    member.role: member.agent_name for member in members
                }
                tasks = [
                    self._bounded_chat(
                        model_id=member.model_id,
                        messages=[
                            ChatMessage(role="system", content=member.system_prompt),
                            ChatMessage(
                                role="user",
                                content=self._council_prompt(member.role, message, prompt_messages),
                            ),
                        ],
                        request_id=f"{request_id}-{member.role}",
                        max_output_tokens=self.settings.deep_mode.council_candidate_tokens,
                    )
                    for member in members
                ]
                responses = await asyncio.gather(*tasks)
                for member, response in zip(members, responses, strict=True):
                    if response is None:
                        warnings.append(
                            f"Council responder {member.role} exceeded the local budget."
                        )
                        continue
                    raw_candidates.append(
                        CouncilCandidate(responder_id=member.role, content=response.content)
                    )
                    for key, value in response.usage.model_dump().items():
                        if isinstance(value, int):
                            usage[key] = usage.get(key, 0) + value
                    warnings.extend(response.warnings)
                metadata["candidate_count"] = len(raw_candidates)
                if raw_candidates:
                    judged = await self._judge_candidates(
                        message=message,
                        candidates=raw_candidates,
                        model_id=model_id,
                        request_id=request_id,
                    )
        except (TimeoutError, AprilError, OSError):
            warnings.append("Council mode exceeded its whole-rung local budget.")
        if not raw_candidates:
            return LadderRun(
                status="unavailable",
                final_message=(
                    "Council mode stopped at the configured local budget before any "
                    "candidate answer was produced."
                ),
                mode="council",
                rung=4,
                model_id=model_id,
                warnings=warnings or ["Council mode produced no complete local candidates."],
                metadata=metadata,
            )
        if judged is not None:
            candidates = judged
            metadata["council_scoring"] = "scout"
        else:
            candidates = [
                score_council_candidate(candidate, question=message) for candidate in raw_candidates
            ]
            metadata["council_scoring"] = "deterministic_rubric"
        best = select_best_candidate(candidates)
        disagreement = _council_disagreement(candidates, best)
        if disagreement is not None:
            metadata["council_disagreement"] = disagreement
        final_message = f"{_MODE_ANNOUNCEMENTS[4]}\n\n{best.content}"
        if disagreement is not None:
            final_message += (
                "\n\nCouncil disagreement: "
                + ", ".join(disagreement["close_responders"])
                + " scored close to the selected answer; treat the choice as contested."
            )
        return LadderRun(
            status="ok",
            final_message=final_message,
            mode="council",
            rung=4,
            model_id=model_id,
            usage=usage,
            warnings=warnings,
            metadata={**metadata, "selected_responder": best.responder_id},
            candidates=candidates,
        )

    async def _judge_candidates(
        self,
        *,
        message: str,
        candidates: list[CouncilCandidate],
        model_id: str,
        request_id: str,
    ) -> list[CouncilCandidate] | None:
        """Score candidates with the local Scout/reading agent when available.

        Returns ``None`` on any failure (runtime down, budget exceeded, invalid
        JSON, unknown responder ids) so the caller falls back to the
        deterministic rubric. Scoring is never faked.
        """
        scout = self.agent_registry.get("reading_agent")
        scout_prompt = (
            scout.system_prompt if scout is not None else "You are APRIL's local scoring agent."
        )
        listing = "\n\n".join(
            f"[{candidate.responder_id}]\n{candidate.content[:1200]}" for candidate in candidates
        )
        response = await self._bounded_chat(
            model_id=scout.model_id if scout is not None and scout.model_id else model_id,
            messages=[
                ChatMessage(role="system", content=scout_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        "Score each candidate answer for correctness, usefulness and "
                        "honesty about uncertainty. Return exactly one JSON object "
                        '{"scores": [{"responder_id": string, "score": number 0..1, '
                        '"rationale": string}]} and nothing else.\n\n'
                        f"Question:\n{message}\n\nCandidates:\n{listing}"
                    ),
                ),
            ],
            request_id=f"{request_id}-judge",
            response_format=ResponseFormat(
                type="json_object",
                json_schema={
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "responder_id": {"type": "string"},
                                    "score": {"type": "number"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["responder_id", "score"],
                            },
                        }
                    },
                    "required": ["scores"],
                },
            ),
            max_output_tokens=self.settings.deep_mode.council_judge_tokens,
        )
        if response is None:
            return None
        try:
            payload = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            return None
        scores = payload.get("scores") if isinstance(payload, dict) else None
        if not isinstance(scores, list):
            return None
        by_id = {candidate.responder_id: candidate for candidate in candidates}
        judged: list[CouncilCandidate] = []
        seen: set[str] = set()
        for item in scores:
            if not isinstance(item, dict):
                continue
            responder_id = str(item.get("responder_id", ""))
            if responder_id not in by_id or responder_id in seen:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            seen.add(responder_id)
            source = by_id[responder_id]
            judged.append(
                CouncilCandidate(
                    responder_id=responder_id,
                    content=source.content,
                    score=max(0.0, min(1.0, score)),
                    rationale=str(item.get("rationale", ""))[:300],
                )
            )
        if len(judged) != len(candidates):
            return None
        return judged

    async def _bounded_chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        request_id: str,
        response_format: ResponseFormat | None = None,
        max_output_tokens: int,
    ) -> Any | None:
        try:
            return await asyncio.wait_for(
                self.runtime_client.chat(
                    model_id=model_id,
                    messages=messages,
                    options=GenerationOptions(max_output_tokens=max_output_tokens),
                    response_format=response_format,
                    request_id=request_id,
                ),
                timeout=self.settings.deep_mode.max_seconds,
            )
        except (TimeoutError, AprilError, OSError):
            return None

    def _can_use_reasoning_mode(self, decision: BrainDecision) -> bool:
        if decision.permission_level > 1 or decision.needs_confirmation:
            return False
        return not decision.tools_needed and not decision.planned_tool_calls

    def _can_use_reflex(self, decision: BrainDecision) -> bool:
        if decision.permission_level != 0 or decision.needs_confirmation:
            return False
        return not decision.tools_needed and not decision.planned_tool_calls

    def _needs_verified_revision(self, message: str, decision: BrainDecision) -> bool:
        if not self._can_use_reasoning_mode(decision):
            return False
        normalized = _normalize(message)
        return any(
            phrase in normalized
            for phrase in {
                "double check",
                "verify your answer",
                "check your answer",
                "critique your answer",
                "revise your answer",
            }
        )

    def _reasoning_system_prompt(self) -> str:
        agent = self.agent_registry.get("reasoning_agent")
        if agent is not None:
            return agent.system_prompt
        return (
            "You are APRIL's local reasoning agent. Answer directly, cite uncertainty, "
            "and do not request tools."
        )

    def _reasoning_user_prompt(
        self,
        *,
        mode: ChatMode,
        message: str,
        prompt_messages: list[ChatMessage],
    ) -> str:
        context = "\n\n".join(item.content for item in prompt_messages if item.role == "user")
        return (
            f"{mode} mode is local-only and read-only. Do not request tools or approvals. "
            "Give a concise answer with assumptions and verification limits.\n\n"
            f"User request:\n{message}\n\nPrepared local context:\n{context}"
        )

    def _council_prompt(
        self,
        role: str,
        message: str,
        prompt_messages: list[ChatMessage],
    ) -> str:
        context = "\n\n".join(item.content for item in prompt_messages if item.role == "user")
        return (
            f"Council responder role: {role}. Work locally and read-only. "
            "Answer the request directly; do not request tools. Include tradeoffs, "
            "tests or validation when relevant, and call out uncertainty.\n\n"
            f"User request:\n{message}\n\nPrepared local context:\n{context}"
        )

    def _agent_model_id(self, agent_name: str, fallback_model_id: str) -> str:
        agent = self.agent_registry.get(agent_name)
        if agent is not None and agent.model_id:
            return agent.model_id
        return fallback_model_id

    def _reasoning_n_council_members(self, model_id: str) -> list[_CouncilMember]:
        """Council seats using the current best-of-N shared reasoning model."""
        return [
            _CouncilMember(
                role=role,
                agent_name=agent_name,
                system_prompt=self._agent_system_prompt(agent_name),
                model_id=model_id,
            )
            for role, agent_name in _COUNCIL_SEATS
        ]

    def _multi_agent_council_members(
        self, *, reasoning_model_id: str, fallback_model_id: str
    ) -> list[_CouncilMember]:
        """Council seats mapped to reasoning/general/creative configured models.

        Prime is the general agent, Sage the reasoning agent, and Muse the
        creative agent. Missing registry entries fall back to the shared
        reasoning prompt/model, and the caller decides whether enough distinct
        models resolved to keep multi-agent mode.
        """
        members: list[_CouncilMember] = []
        for role, agent_name in _COUNCIL_SEATS:
            if agent_name == "reasoning_agent":
                member_model_id = reasoning_model_id
            else:
                member_model_id = self._agent_model_id(agent_name, fallback_model_id)
            members.append(
                _CouncilMember(
                    role=role,
                    agent_name=agent_name,
                    system_prompt=self._agent_system_prompt(agent_name),
                    model_id=member_model_id,
                )
            )
        return members

    def _agent_system_prompt(self, agent_name: str) -> str:
        agent = self.agent_registry.get(agent_name)
        if agent is not None:
            return agent.system_prompt
        return self._reasoning_system_prompt()


_COUNCIL_SEATS: tuple[tuple[str, str], ...] = (
    ("prime", "general_agent"),
    ("sage", "reasoning_agent"),
    ("muse", "creative_agent"),
)


def score_council_candidate(candidate: CouncilCandidate, *, question: str) -> CouncilCandidate:
    content = candidate.content.strip()
    normalized = content.lower()
    question_terms = set(re.findall(r"[a-z0-9_]{4,}", question.lower()))
    overlap = sum(1 for term in question_terms if term in normalized)
    score = min(len(content), 900) / 900
    score += min(overlap, 5) * 0.4
    for marker in ("because", "tradeoff", "trade-off", "risk", "test", "verify", "local"):
        if marker in normalized:
            score += 0.25
    if len(content) < 40:
        score -= 0.5
    if "i can't" in normalized or "cannot" in normalized:
        score -= 0.1
    rationale = f"overlap={overlap}; chars={len(content)}"
    return CouncilCandidate(
        responder_id=candidate.responder_id,
        content=content,
        score=round(score, 4),
        rationale=rationale,
    )


def select_best_candidate(candidates: list[CouncilCandidate]) -> CouncilCandidate:
    if not candidates:
        raise ValueError("No council candidates to score.")
    return max(candidates, key=lambda candidate: (candidate.score, len(candidate.content)))


def _budget_unresolved_model_metadata(fallback_model_id: str) -> dict[str, str]:
    return {
        "requested_role": "reasoning",
        "selected_role": "brain",
        "selected_model_id": fallback_model_id,
        "fallback_model_id": fallback_model_id,
        "reason": "not_resolved_before_budget",
    }


_DISAGREEMENT_SCORE_MARGIN = 0.15


def _council_disagreement(
    candidates: list[CouncilCandidate], best: CouncilCandidate
) -> dict[str, Any] | None:
    """Surface near-ties: responders whose score is within the margin of best."""
    close = [
        candidate.responder_id
        for candidate in candidates
        if candidate.responder_id != best.responder_id
        and best.score - candidate.score <= _DISAGREEMENT_SCORE_MARGIN
    ]
    if not close:
        return None
    return {
        "selected": best.responder_id,
        "close_responders": close,
        "scores": {candidate.responder_id: candidate.score for candidate in candidates},
    }


def _normalize(message: str) -> str:
    normalized = message.strip().lower()
    normalized = re.sub(r"^(?:april|hey april|ok april)[,\s]+", "", normalized)
    normalized = re.sub(r"[?.!]+$", "", normalized)
    return " ".join(normalized.split())
