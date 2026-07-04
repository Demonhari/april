from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from april_common.settings import AprilSettings
from services.memory.sqlite_memory import SqliteMemory

# Only a small number of overlay candidates per night keeps evolution reviewable.
MAX_CANDIDATES_PER_RUN = 2


@dataclass(frozen=True, slots=True)
class OverlayCandidate:
    agent: str
    content: str
    source_summary: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "content": self.content,
            "source_summary": self.source_summary,
        }


async def generate_overlay_candidates(
    memory: SqliteMemory,
    settings: AprilSettings,
) -> list[OverlayCandidate]:
    """D4: derive bounded prompt-overlay candidates from local feedback — as data.

    The generator is deterministic: negative feedback reasons are grouped by
    the agent that produced the run and rendered through a fixed template.
    Candidates are plain advisory prose bounded by the configured overlay
    budget; they never touch repo prompts and only become active if they pass
    the D5 eval gate (and approval, for write-capable agents).
    """
    reasons_by_agent: dict[str, list[str]] = {}
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
        reasons = reasons_by_agent.setdefault(agent, [])
        normalized = " ".join(event.reason.split())
        if normalized and normalized not in reasons:
            reasons.append(normalized)

    candidates: list[OverlayCandidate] = []
    budget = settings.evolution.prompt_overlay_max_chars
    for agent in sorted(reasons_by_agent):
        if len(candidates) >= MAX_CANDIDATES_PER_RUN:
            break
        reasons = reasons_by_agent[agent][:5]
        body = (
            "Recent local user feedback flagged these issues:\n"
            + "\n".join(f"- {reason[:200]}" for reason in reasons)
            + "\nWhen similar requests appear, address these concerns explicitly "
            "and say what you changed."
        )
        if budget > 0:
            body = body[:budget]
        candidates.append(
            OverlayCandidate(
                agent=agent,
                content=body,
                source_summary=f"{len(reasons)} negative feedback reason(s)",
            )
        )
    return candidates
