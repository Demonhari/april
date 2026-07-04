from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from skills.playbooks.schema import PlaybookDefinition, PlaybookStep


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
    ) -> list[PlaybookDefinition]:
        """Mine deterministic candidates from repeated successful sequences."""

        existing = existing_ids or set()
        grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
        for calls in sequences:
            successful = _successful_calls(calls, known_tools=known_tools)
            if len(successful) < 2:
                continue
            signature = _sequence_signature(successful)
            grouped[signature].append(successful)

        candidates: list[PlaybookDefinition] = []
        for signature in sorted(grouped):
            supported = grouped[signature]
            if len(supported) < support_threshold:
                continue
            digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
            candidate_id = f"mined-{digest}"
            if candidate_id in existing:
                continue
            steps = [
                PlaybookStep(
                    tool=str(call["tool"]),
                    args=dict(call["args"]),
                    reason=(
                        "Mined from a frequent successful local tool sequence "
                        f"(support={len(supported)})."
                    ),
                    agent_id=call.get("agent") if isinstance(call.get("agent"), str) else None,
                )
                for call in supported[0][:20]
            ]
            candidates.append(
                PlaybookDefinition(
                    id=candidate_id,
                    name=f"Mined frequent sequence {digest}",
                    description=(
                        "Candidate mined from a frequent successful local tool sequence "
                        f"observed {len(supported)} times."
                    ),
                    status="candidate",
                    trigger_examples=[],
                    steps=steps,
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


def _sequence_signature(tool_calls: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "tool": str(call["tool"]),
                "args": dict(call["args"]),
                "agent": call.get("agent") if isinstance(call.get("agent"), str) else None,
            }
            for call in tool_calls[:20]
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
