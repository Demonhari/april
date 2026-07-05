from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from skills.playbooks.schema import PlaybookDefinition, PlaybookStep

# Bounds keep subsequence enumeration cheap and candidates reviewable.
MIN_CANDIDATE_STEPS = 2
MAX_CANDIDATE_STEPS = 10
MAX_CANDIDATES = 10
_MAX_SEQUENCE_CALLS = 30


@dataclass(slots=True)
class MinedCandidate:
    """A frequent mined subsequence plus where it was observed."""

    definition: PlaybookDefinition
    support: int
    sequence_indexes: list[int] = field(default_factory=list)


class PlaybookMiner:
    def mine(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        name: str = "Mined playbook",
        trigger: str = "",
    ) -> PlaybookDefinition | None:
        successful = [
            call
            for call in tool_calls
            if call.get("status") in {"executed", "ok", "success"}
            and isinstance(call.get("tool"), str)
            and isinstance(call.get("args"), dict)
        ]
        if len(successful) < 2:
            return None
        steps = [
            PlaybookStep(
                tool=str(call["tool"]),
                args=dict(call["args"]),
                reason="Mined from a successful local tool sequence.",
                agent_id=call.get("agent") if isinstance(call.get("agent"), str) else None,
            )
            for call in successful[:20]
        ]
        digest = hashlib.sha256(
            "|".join(f"{step.tool}:{sorted(step.args)}" for step in steps).encode("utf-8")
        ).hexdigest()[:12]
        return PlaybookDefinition(
            id=f"mined-{digest}",
            name=name,
            description="Candidate mined from successful local tool calls.",
            status="candidate",
            source="learned",
            trigger_examples=[trigger] if trigger else [],
            steps=steps,
        )

    def mine_frequent(
        self,
        sequences: list[list[dict[str, Any]]],
        *,
        support_threshold: int = 3,
        existing_ids: set[str] | None = None,
        known_tools: set[str] | None = None,
        min_steps: int = MIN_CANDIDATE_STEPS,
        max_steps: int = MAX_CANDIDATE_STEPS,
        max_candidates: int = MAX_CANDIDATES,
    ) -> list[PlaybookDefinition]:
        """Mine deterministic candidates from repeated successful sequences."""
        return [
            candidate.definition
            for candidate in self.mine_frequent_detailed(
                sequences,
                support_threshold=support_threshold,
                existing_ids=existing_ids,
                known_tools=known_tools,
                min_steps=min_steps,
                max_steps=max_steps,
                max_candidates=max_candidates,
            )
        ]

    def mine_frequent_detailed(
        self,
        sequences: list[list[dict[str, Any]]],
        *,
        support_threshold: int = 3,
        existing_ids: set[str] | None = None,
        known_tools: set[str] | None = None,
        min_steps: int = MIN_CANDIDATE_STEPS,
        max_steps: int = MAX_CANDIDATE_STEPS,
        max_candidates: int = MAX_CANDIDATES,
    ) -> list[MinedCandidate]:
        """Mine frequent *contiguous subsequences* of successful tool calls.

        Support counts how many distinct input sequences (conversations)
        contain the subsequence at least once. Subsequences strictly contained
        in a longer frequent subsequence with at least the same support are
        suppressed (closed patterns), so a repeated three-step flow yields one
        three-step candidate, not three overlapping fragments. Candidate size,
        step count, and candidate count are all bounded.
        """
        existing = existing_ids or set()
        min_steps = max(2, min_steps)
        max_steps = max(min_steps, max_steps)
        occurrences: dict[str, set[int]] = defaultdict(set)
        calls_by_signature: dict[str, list[dict[str, Any]]] = {}
        for index, calls in enumerate(sequences):
            successful = _successful_calls(calls, known_tools=known_tools)
            if len(successful) < min_steps:
                continue
            bounded = successful[:_MAX_SEQUENCE_CALLS]
            for start in range(len(bounded)):
                for length in range(min_steps, max_steps + 1):
                    end = start + length
                    if end > len(bounded):
                        break
                    window = bounded[start:end]
                    signature = _sequence_signature(window)
                    occurrences[signature].add(index)
                    calls_by_signature.setdefault(signature, window)

        frequent = [
            (signature, calls_by_signature[signature], indexes)
            for signature, indexes in occurrences.items()
            if len(indexes) >= support_threshold
        ]
        tokens_by_signature = {
            signature: [_call_signature(call) for call in calls]
            for signature, calls, _indexes in frequent
        }
        kept: list[tuple[str, list[dict[str, Any]], set[int]]] = []
        for signature, calls, indexes in frequent:
            tokens = tokens_by_signature[signature]
            dominated = any(
                other_signature != signature
                and len(tokens_by_signature[other_signature]) > len(tokens)
                and len(other_indexes) >= len(indexes)
                and _contains_contiguous(tokens_by_signature[other_signature], tokens)
                for other_signature, _other_calls, other_indexes in frequent
            )
            if not dominated:
                kept.append((signature, calls, indexes))

        # Deterministic order: longest first, most support first, then signature.
        kept.sort(key=lambda item: (-len(item[1]), -len(item[2]), item[0]))
        candidates: list[MinedCandidate] = []
        for signature, calls, indexes in kept:
            if len(candidates) >= max_candidates:
                break
            digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
            candidate_id = f"mined-{digest}"
            if candidate_id in existing:
                continue
            support = len(indexes)
            steps = [
                PlaybookStep(
                    tool=str(call["tool"]),
                    args=dict(call["args"]),
                    reason=(
                        f"Mined from a frequent successful local tool sequence (support={support})."
                    ),
                    agent_id=call.get("agent") if isinstance(call.get("agent"), str) else None,
                )
                for call in calls
            ]
            candidates.append(
                MinedCandidate(
                    definition=PlaybookDefinition(
                        id=candidate_id,
                        name=f"Mined frequent sequence {digest}",
                        description=(
                            "Candidate mined from a frequent successful local tool "
                            f"sequence observed in {support} conversations."
                        ),
                        status="candidate",
                        source="learned",
                        trigger_examples=[],
                        steps=steps,
                    ),
                    support=support,
                    sequence_indexes=sorted(indexes),
                )
            )
        return candidates


def _successful_calls(
    tool_calls: list[dict[str, Any]],
    *,
    known_tools: set[str] | None,
) -> list[dict[str, Any]]:
    successful: list[dict[str, Any]] = []
    for call in tool_calls:
        tool = call.get("tool")
        args = call.get("args")
        if call.get("status") not in {"executed", "ok", "success"}:
            continue
        if not isinstance(tool, str) or not isinstance(args, dict):
            continue
        if known_tools is not None and tool not in known_tools:
            return []
        successful.append(call)
    return successful


def _call_signature(call: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool": str(call["tool"]),
            "args": dict(call["args"]),
            "agent": call.get("agent") if isinstance(call.get("agent"), str) else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _sequence_signature(tool_calls: list[dict[str, Any]]) -> str:
    return "[" + ",".join(_call_signature(call) for call in tool_calls[:20]) + "]"


def _contains_contiguous(longer: list[str], shorter: list[str]) -> bool:
    if len(shorter) > len(longer):
        return False
    span = len(shorter)
    return any(longer[start : start + span] == shorter for start in range(len(longer) - span + 1))
