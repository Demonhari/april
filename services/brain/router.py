from __future__ import annotations

from april_common.errors import AprilError
from services.april_runtime.client import RuntimeClient
from services.april_runtime.schemas import ChatMessage
from services.brain.fallback_router import FallbackRouter
from services.brain.parser import parse_with_repair
from services.brain.schemas import BrainDecision

ROUTER_SYSTEM_PROMPT = """Route this request. Return exactly one JSON object matching:
{"intent": "...", "agent": "...", "model_id": "...", "tools_needed": [],
"planned_tool_calls": [{"tool": "...", "args": {}, "reason": "..."}], "memory_queries": [],
"permission_level": 0, "risk_level": "none", "needs_confirmation": false,
"task_steps": ["short operational step"], "decision_summary": "concise summary"}.
Do not include chain-of-thought."""


class BrainRouter:
    def __init__(
        self, runtime_client: RuntimeClient, *, brain_model_id: str = "april-brain"
    ) -> None:
        self.runtime_client = runtime_client
        self.brain_model_id = brain_model_id
        self.fallback = FallbackRouter()

    async def route(self, message: str, *, request_id: str | None = None) -> BrainDecision:
        try:
            response = await self.runtime_client.chat(
                model_id=self.brain_model_id,
                messages=[
                    ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=message),
                ],
                request_id=request_id,
            )

            async def repair(_: str) -> str:
                repaired = await self.runtime_client.chat(
                    model_id=self.brain_model_id,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=(
                                "Repair the previous response into exactly one "
                                "valid JSON object."
                            ),
                        ),
                        ChatMessage(role="user", content=response.content),
                    ],
                    request_id=request_id,
                )
                return repaired.content

            return await parse_with_repair(response.content, repair)
        except AprilError:
            return self.fallback.route(message)
