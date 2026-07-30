from __future__ import annotations

from typing import Literal

StreamEventName = Literal[
    "meta",
    "routing",
    "agent_iteration",
    "tool_request",
    "tool_result",
    "approval_required",
    "final_answer",
    "token",
    "usage",
    "done",
    "error",
]
