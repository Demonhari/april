from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from agents.registry import AgentRegistry
from april_common.settings import AprilSettings
from services.april_runtime.client import RuntimeClient
from services.april_runtime.schemas import ChatMessage, GenerationOptions, ResponseFormat
from services.brain.reasoning_resolver import resolve_reasoning_model
from services.brain.schemas import BrainDecision

ChatMode = Literal["standard", "deep", "council"]
LadderStatus = Literal["ok", "unavailable"]

_MODE_ANNOUNCEMENTS: dict[int, str] = {
    0: "Mode: reflex (local deterministic answer).",
    2: "Mode: verified (local self-check).",
    3: "Mode: deep (local reasoning).",
    4: "Mode: council (local best-of-N).",
}

# Confidence-driven rung thresholds: routing confidence below DEEP goes to the
# deep rung, between DEEP and VERIFIED to the verified rung, above VERIFIED to
# the standard path. All deterministic; tool/approval paths always win.
DEEP_CONFIDENCE_THRESHOLD = 0.4
VERIFIED_CONFIDENCE_THRESHOLD = 0.7

_DEEP_PHRASES = (
    "/deep",
    "think hard",
    "think harder",
    "think deeply",
    "think carefully",
    "reason step by step",
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
    ) -> LadderSelection:
        high_stakes = self.is_high_stakes(message)
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
        if decision.confidence < DEEP_CONFIDENCE_THRESHOLD:
            return LadderSelection(
                mode=mode,
                rung=3,
                reason=(
                    f"routing confidence {decision.confidence:.2f} below "
                    f"{DEEP_CONFIDENCE_THRESHOLD}"
                ),
                announcement=_MODE_ANNOUNCEMENTS[3],
            )
        if decision.confidence < VERIFIED_CONFIDENCE_THRESHOLD:
            return LadderSelection(
                mode=mode,
                rung=2,
                reason=(
                    f"routing confidence {decision.confidence:.2f} below "
                    f"{VERIFIED_CONFIDENCE_THRESHOLD}"
                ),
                announcement=_MODE_ANNOUNCEMENTS[2],
            )
        return LadderSelection(mode=mode, rung=1, reason="standard route")

    def is_high_stakes(self, message: str) -> bool:
        """Deterministic high-stakes tagging from explicit phrases only."""
        normalized = _normalize(message)
        return any(phrase in normalized for phrase in _HIGH_STAKES_PHRASES)

    def is_reflex_query(self, message: str, decision: BrainDecision) -> bool:
        if decision.permission_level != 0 or decision.tools_needed or decision.planned_tool_calls:
            return False
        normalized = _normalize(message)
        return normalized in {
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
        resolution = await resolve_reasoning_model(
            runtime_client=self.runtime_client,
            fallback_model_id=fallback_model_id,
        )
        user_prompt = self._reasoning_user_prompt(
            mode="deep",
            message=message,
            prompt_messages=prompt_messages,
        )
        response = await self._bounded_chat(
            model_id=resolution.model_id,
            messages=[
                ChatMessage(role="system", content=self._reasoning_system_prompt()),
                ChatMessage(role="user", content=user_prompt),
            ],
            request_id=request_id,
        )
        metadata = {
            "mode": "deep",
            "intelligence_rung": 3,
            "budget_seconds": self.settings.deep_mode.max_seconds,
            "model_resolution": resolution.metadata(),
        }
        if response is None:
            return LadderRun(
                status="unavailable",
                final_message=(
                    "Deep mode stopped at the configured local budget before a complete "
                    "answer was produced."
                ),
                mode="deep",
                rung=3,
                model_id=resolution.model_id,
                warnings=["Deep mode exceeded its configured local budget."],
                metadata=metadata,
            )
        return LadderRun(
            status="ok",
            final_message=f"{_MODE_ANNOUNCEMENTS[3]}\n\n{response.content}",
            mode="deep",
            rung=3,
            model_id=resolution.model_id,
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
        response = await self._bounded_chat(
            model_id=model_id,
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are APRIL's local verifier. Return exactly one JSON object. "
                        "Do not include hidden reasoning."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Check the assistant answer for correctness, unsupported claims, "
                        "and missing caveats. Return "
                        '{"needs_revision": boolean, "final_answer": string, "reason": string}.\n'
                        f"User request:\n{message}\n\nAssistant answer:\n{initial_answer}"
                    ),
                ),
            ],
            request_id=request_id,
            response_format=ResponseFormat(
                type="json_object",
                json_schema={
                    "type": "object",
                    "properties": {
                        "needs_revision": {"type": "boolean"},
                        "final_answer": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["needs_revision", "final_answer", "reason"],
                },
            ),
        )
        metadata = {
            "mode": "standard",
            "intelligence_rung": 2,
            "budget_seconds": self.settings.deep_mode.max_seconds,
        }
        if response is None:
            return LadderRun(
                status="ok",
                final_message=initial_answer,
                mode="standard",
                rung=2,
                model_id=model_id,
                warnings=["Verification exceeded its configured local budget."],
                metadata=metadata,
            )
        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError:
            return LadderRun(
                status="ok",
                final_message=initial_answer,
                mode="standard",
                rung=2,
                model_id=model_id,
                usage=response.usage.model_dump(),
                warnings=["Verification returned invalid JSON; kept the original answer."],
                metadata=metadata,
            )
        final_answer = str(payload.get("final_answer") or initial_answer)
        return LadderRun(
            status="ok",
            final_message=f"{_MODE_ANNOUNCEMENTS[2]}\n\n{final_answer}",
            mode="standard",
            rung=2,
            model_id=model_id,
            usage=response.usage.model_dump(),
            warnings=response.warnings,
            metadata={**metadata, "verification_reason": str(payload.get("reason") or "")},
        )

    async def run_council(
        self,
        *,
        message: str,
        prompt_messages: list[ChatMessage],
        fallback_model_id: str,
        request_id: str,
    ) -> LadderRun:
        resolution = await resolve_reasoning_model(
            runtime_client=self.runtime_client,
            fallback_model_id=fallback_model_id,
        )
        members = self._council_members()[: self.settings.deep_mode.council_n]
        tasks = [
            self._bounded_chat(
                model_id=resolution.model_id,
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(
                        role="user",
                        content=self._council_prompt(role, message, prompt_messages),
                    ),
                ],
                request_id=f"{request_id}-{role}",
            )
            for role, system_prompt in members
        ]
        responses = await asyncio.gather(*tasks)
        raw_candidates: list[CouncilCandidate] = []
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        warnings: list[str] = []
        for (role, _prompt), response in zip(members, responses, strict=True):
            if response is None:
                warnings.append(f"Council responder {role} exceeded the local budget.")
                continue
            raw_candidates.append(CouncilCandidate(responder_id=role, content=response.content))
            for key, value in response.usage.model_dump().items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
            warnings.extend(response.warnings)
        metadata: dict[str, Any] = {
            "mode": "council",
            "intelligence_rung": 4,
            "budget_seconds": self.settings.deep_mode.max_seconds,
            "model_resolution": resolution.metadata(),
            "candidate_count": len(raw_candidates),
            "council_members": [role for role, _prompt in members],
        }
        if not raw_candidates:
            return LadderRun(
                status="unavailable",
                final_message=(
                    "Council mode stopped at the configured local budget before any "
                    "candidate answer was produced."
                ),
                mode="council",
                rung=4,
                model_id=resolution.model_id,
                warnings=warnings or ["Council mode produced no complete local candidates."],
                metadata=metadata,
            )
        judged = await self._judge_candidates(
            message=message,
            candidates=raw_candidates,
            model_id=resolution.model_id,
            request_id=request_id,
        )
        if judged is not None:
            candidates = judged
            metadata["council_scoring"] = "scout"
        else:
            candidates = [
                score_council_candidate(candidate, question=message)
                for candidate in raw_candidates
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
            model_id=resolution.model_id,
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
            scout.system_prompt
            if scout is not None
            else "You are APRIL's local scoring agent."
        )
        listing = "\n\n".join(
            f"[{candidate.responder_id}]\n{candidate.content[:1200]}"
            for candidate in candidates
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
    ) -> Any | None:
        try:
            return await asyncio.wait_for(
                self.runtime_client.chat(
                    model_id=model_id,
                    messages=messages,
                    options=GenerationOptions(max_output_tokens=1536),
                    response_format=response_format,
                    request_id=request_id,
                ),
                timeout=self.settings.deep_mode.max_seconds,
            )
        except TimeoutError:
            return None

    def _can_use_reasoning_mode(self, decision: BrainDecision) -> bool:
        if decision.permission_level > 1 or decision.needs_confirmation:
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

    def _council_members(self) -> list[tuple[str, str]]:
        """Council seats mapped to the architecture's agent prompts.

        Prime is the general agent, Sage the reasoning agent, and Muse the
        creative agent. Missing registry entries fall back to the reasoning
        system prompt so the council always has its configured seat count.
        """
        seats = (
            ("prime", "general_agent"),
            ("sage", "reasoning_agent"),
            ("muse", "creative_agent"),
        )
        members: list[tuple[str, str]] = []
        for role, agent_name in seats:
            agent = self.agent_registry.get(agent_name)
            prompt = agent.system_prompt if agent is not None else self._reasoning_system_prompt()
            members.append((role, prompt))
        return members


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
