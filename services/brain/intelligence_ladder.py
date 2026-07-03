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


@dataclass(frozen=True, slots=True)
class LadderSelection:
    mode: ChatMode
    rung: int
    reason: str
    announcement: str | None = None


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
        if self.is_reflex_query(message, decision):
            return LadderSelection(
                mode=mode,
                rung=0,
                reason="trivial local lookup",
                announcement=_MODE_ANNOUNCEMENTS[0],
            )
        if mode == "council":
            if self._can_use_reasoning_mode(decision):
                return LadderSelection(
                    mode=mode,
                    rung=4,
                    reason="explicit council mode",
                    announcement=_MODE_ANNOUNCEMENTS[4],
                )
            return LadderSelection(
                mode=mode,
                rung=1,
                reason="tool or approval path uses the standard permission flow",
            )
        if mode == "deep":
            if self._can_use_reasoning_mode(decision):
                return LadderSelection(
                    mode=mode,
                    rung=3,
                    reason="explicit deep mode",
                    announcement=_MODE_ANNOUNCEMENTS[3],
                )
            return LadderSelection(
                mode=mode,
                rung=1,
                reason="tool or approval path uses the standard permission flow",
            )
        if self._needs_verified_revision(message, decision):
            return LadderSelection(
                mode=mode,
                rung=2,
                reason="explicit verification request",
                announcement=_MODE_ANNOUNCEMENTS[2],
            )
        return LadderSelection(mode=mode, rung=1, reason="standard route")

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
        roles = self._council_roles()[: self.settings.deep_mode.council_n]
        tasks = [
            self._bounded_chat(
                model_id=resolution.model_id,
                messages=[
                    ChatMessage(role="system", content=self._reasoning_system_prompt()),
                    ChatMessage(
                        role="user",
                        content=self._council_prompt(role, message, prompt_messages),
                    ),
                ],
                request_id=f"{request_id}-{role}",
            )
            for role in roles
        ]
        responses = await asyncio.gather(*tasks)
        candidates: list[CouncilCandidate] = []
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        warnings: list[str] = []
        for role, response in zip(roles, responses, strict=True):
            if response is None:
                warnings.append(f"Council responder {role} exceeded the local budget.")
                continue
            candidate = score_council_candidate(
                CouncilCandidate(responder_id=role, content=response.content),
                question=message,
            )
            candidates.append(candidate)
            for key, value in response.usage.model_dump().items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
            warnings.extend(response.warnings)
        metadata = {
            "mode": "council",
            "intelligence_rung": 4,
            "budget_seconds": self.settings.deep_mode.max_seconds,
            "model_resolution": resolution.metadata(),
            "candidate_count": len(candidates),
        }
        if not candidates:
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
        best = select_best_candidate(candidates)
        return LadderRun(
            status="ok",
            final_message=(
                f"{_MODE_ANNOUNCEMENTS[4]}\n\n"
                f"{best.content}"
            ),
            mode="council",
            rung=4,
            model_id=resolution.model_id,
            usage=usage,
            warnings=warnings,
            metadata={**metadata, "selected_responder": best.responder_id},
            candidates=candidates,
        )

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

    def _council_roles(self) -> list[str]:
        return ["direct", "skeptical", "practical", "risk"]


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


def _normalize(message: str) -> str:
    normalized = message.strip().lower()
    normalized = re.sub(r"^(?:april|hey april|ok april)[,\s]+", "", normalized)
    normalized = re.sub(r"[?.!]+$", "", normalized)
    return " ".join(normalized.split())
