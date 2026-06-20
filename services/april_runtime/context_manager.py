from __future__ import annotations

from dataclasses import dataclass

from services.april_runtime.backend import RuntimeBackend
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.schemas import ChatMessage


@dataclass(frozen=True, slots=True)
class ContextResult:
    messages: list[ChatMessage]
    truncated: bool
    input_tokens: int


class ContextManager:
    async def fit(
        self,
        *,
        model: ModelDefinition,
        backend: RuntimeBackend,
        messages: list[ChatMessage],
        max_output_tokens: int,
    ) -> ContextResult:
        budget = max(model.context_size - max_output_tokens, 1)
        system_messages = [message for message in messages if message.role == "system"]
        other_messages = [message for message in messages if message.role != "system"]
        selected: list[ChatMessage] = []
        total = 0
        for message in system_messages:
            tokens = await backend.count_tokens(message.content)
            total += tokens
            selected.append(message)
        truncated = False
        newest: list[ChatMessage] = []
        for message in reversed(other_messages):
            tokens = await backend.count_tokens(message.content)
            if total + tokens > budget:
                truncated = True
                continue
            newest.append(message)
            total += tokens
        selected.extend(reversed(newest))
        return ContextResult(messages=selected, truncated=truncated, input_tokens=total)
