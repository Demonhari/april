from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from april_common.audit import AuditLogger
from april_common.errors import AprilError
from april_common.settings import AprilSettings
from services.april_runtime.schemas import ChatMessage, ChatResponse, GenerationOptions
from services.evolution.versions import prompt_overlay_rejection_reason
from services.memory.schemas import MemoryRecord
from services.memory.sqlite_memory import SqliteMemory

# Only a small number of overlay candidates per night keeps evolution reviewable.
MAX_CANDIDATES_PER_RUN = 2
_SOURCE_SESSION_RE = re.compile(r"\s*\(source_session=([^()]+)\)\s*$")
_WINNER_RE = re.compile(r"(?:^|\s)winner=([^\s]+)")


class OverlayDraftRuntimeClient(Protocol):
    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
        request_id: str | None = None,
    ) -> ChatResponse: ...


@dataclass(frozen=True, slots=True)
class OverlayCandidate:
    agent: str
    content: str
    source_summary: str
    tier: Literal["deterministic", "model_drafted"] = "deterministic"

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "content": self.content,
            "source_summary": self.source_summary,
            "tier": self.tier,
        }


@dataclass(frozen=True, slots=True)
class _GuidanceEvidence:
    agent: str
    line: str
    created_at: str
    confidence: float
    source: Literal["correction", "contradiction", "negative_feedback"]


async def generate_overlay_candidates(
    memory: SqliteMemory,
    settings: AprilSettings,
    *,
    runtime_client: OverlayDraftRuntimeClient | None = None,
    audit: AuditLogger | None = None,
) -> list[OverlayCandidate]:
    """Synthesize bounded advisory overlays from local, inspectable evidence.

    Tier A is deterministic and always enabled. Tier B reserves one of the
    existing candidate slots when configured, but only emits bytes returned by
    the injected local Runtime client. Every candidate remains subject to the
    same D5 ratchet and PromptOverlayManager approval/load checks.
    """
    evidence = await _collect_evidence(memory)
    tier_a = _deterministic_candidates(
        evidence,
        max_chars=settings.evolution.prompt_overlay_max_chars,
    )
    model_enabled = settings.evolution.model_drafted_overlays
    deterministic_limit = MAX_CANDIDATES_PER_RUN - (1 if model_enabled else 0)
    candidates = tier_a[:deterministic_limit]
    if not model_enabled:
        return candidates
    if not tier_a:
        _audit_model_draft(audit, "model_drafted_overlay_skipped", "no learned inputs")
        return candidates
    if runtime_client is None:
        _audit_model_draft(
            audit,
            "model_drafted_overlay_skipped",
            "local runtime client unavailable",
        )
        return candidates

    seed = tier_a[0]
    try:
        response = await runtime_client.chat(
            model_id=settings.brain.model_id,
            messages=_draft_messages(seed),
            options=GenerationOptions(temperature=0.2, max_output_tokens=256),
            request_id="dreamer-model-drafted-overlay",
        )
    except (AprilError, OSError) as exc:
        _audit_model_draft(
            audit,
            "model_drafted_overlay_skipped",
            f"local runtime unavailable: {exc}",
        )
        return candidates

    drafted = response.content.strip()
    reason = prompt_overlay_rejection_reason(
        drafted,
        max_chars=settings.evolution.prompt_overlay_max_chars,
    )
    if not drafted:
        reason = "model returned an empty draft"
    if reason is not None:
        _audit_model_draft(audit, "model_drafted_overlay_rejected", reason)
        return candidates
    if any(_normalized(candidate.content) == _normalized(drafted) for candidate in candidates):
        _audit_model_draft(audit, "model_drafted_overlay_rejected", "duplicate candidate")
        return candidates
    candidates.append(
        OverlayCandidate(
            agent=seed.agent,
            content=drafted,
            source_summary="Archive model draft from deterministic learned inputs",
            tier="model_drafted",
        )
    )
    return candidates[:MAX_CANDIDATES_PER_RUN]


async def _collect_evidence(memory: SqliteMemory) -> list[_GuidanceEvidence]:
    evidence: list[_GuidanceEvidence] = []
    for record in await memory.list_memories():
        if record.kind != "correction" or record.source not in {
            "reflection",
            "archive",
            "dream",
        }:
            continue
        context = _SOURCE_SESSION_RE.sub("", record.reason).strip()
        if not context:
            context = "a similar situation is observed"
        evidence.append(
            _GuidanceEvidence(
                agent=await _originating_agent(memory, record),
                line=f"When {_without_terminal_punctuation(context)}, {_sentence(record.content)}",
                created_at=record.created_at,
                confidence=record.confidence,
                source="correction",
            )
        )

    for pair in await memory.list_memory_contradictions(status="resolved"):
        match = _WINNER_RE.search(pair.resolution or "")
        if match is None:
            continue
        winner = await memory.get_memory(match.group(1), include_inactive=True)
        if winner is None or winner.superseded_by is not None:
            continue
        evidence.append(
            _GuidanceEvidence(
                agent=await _originating_agent(memory, winner),
                line=f"Treat this as the surviving fact: {_sentence(winner.content)}",
                created_at=pair.resolved_at or pair.created_at,
                confidence=winner.confidence,
                source="contradiction",
            )
        )

    for event in await memory.list_feedback_events(limit=100):
        if event.rating != "bad" or not event.reason:
            continue
        agent = "general_agent"
        if event.agent_run_id is not None:
            row = await memory.database.fetchone(
                "SELECT agent FROM agent_runs WHERE id = ?", (event.agent_run_id,)
            )
            if row is not None and row["agent"]:
                agent = str(row["agent"])
        reason = " ".join(event.reason.split())
        if reason:
            evidence.append(
                _GuidanceEvidence(
                    agent=agent,
                    line=f"Address recent negative feedback: {_sentence(reason)}",
                    created_at=event.created_at,
                    confidence=0.0,
                    source="negative_feedback",
                )
            )
    return sorted(
        evidence,
        key=lambda item: (item.created_at, item.confidence, item.agent, item.line),
        reverse=True,
    )


async def _originating_agent(memory: SqliteMemory, record: MemoryRecord) -> str:
    match = _SOURCE_SESSION_RE.search(record.reason)
    if match is None:
        return "general_agent"
    row = await memory.database.fetchone(
        """
        SELECT agent_runs.agent
        FROM sessions
        JOIN agent_runs ON agent_runs.conversation_id = sessions.conversation_id
        WHERE sessions.id = ?
        ORDER BY agent_runs.created_at DESC, agent_runs.id DESC
        LIMIT 1
        """,
        (match.group(1),),
    )
    if row is None or not row["agent"]:
        return "general_agent"
    return str(row["agent"])


def _deterministic_candidates(
    evidence: list[_GuidanceEvidence], *, max_chars: int
) -> list[OverlayCandidate]:
    by_agent: dict[str, list[_GuidanceEvidence]] = {}
    seen_by_agent: dict[str, set[str]] = {}
    for item in evidence:
        normalized = _normalized(item.line)
        seen = seen_by_agent.setdefault(item.agent, set())
        if normalized in seen:
            continue
        seen.add(normalized)
        by_agent.setdefault(item.agent, []).append(item)

    candidates: list[OverlayCandidate] = []
    for agent, items in by_agent.items():
        body = "Learned local guidance:\n" + "\n".join(f"- {item.line}" for item in items[:8])
        if max_chars > 0:
            body = body[:max_chars]
        if prompt_overlay_rejection_reason(body, max_chars=max_chars) is not None:
            continue
        source_counts = {
            source: sum(1 for item in items if item.source == source)
            for source in ("correction", "contradiction", "negative_feedback")
        }
        candidates.append(
            OverlayCandidate(
                agent=agent,
                content=body,
                source_summary=", ".join(
                    f"{count} {source.replace('_', ' ')}"
                    for source, count in source_counts.items()
                    if count
                ),
            )
        )
    return candidates


def _draft_messages(seed: OverlayCandidate) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You are Archive, APRIL's memory specialist. Draft exactly one short prompt "
                "overlay for the named agent using only the supplied learned inputs. Return "
                "strict advisory prose only. Do not state or alter tools, permissions, policy, "
                "or executable actions. Do not use headings, YAML, JSON, or markdown fences."
            ),
        ),
        ChatMessage(
            role="user",
            content=f"Target agent: {seed.agent}\nLearned inputs:\n{seed.content}",
        ),
    ]


def _sentence(value: str) -> str:
    text = " ".join(value.split())
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _without_terminal_punctuation(value: str) -> str:
    return " ".join(value.split()).rstrip(".!?")


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _audit_model_draft(audit: AuditLogger | None, event_type: str, reason: str) -> None:
    if audit is None:
        return
    audit.write(
        {
            "event_type": event_type,
            "actor": "archive_agent",
            "reason": reason[:500],
        }
    )
