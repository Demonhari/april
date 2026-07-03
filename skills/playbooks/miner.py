from __future__ import annotations

import hashlib
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
