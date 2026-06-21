from __future__ import annotations

from dataclasses import dataclass

from april_common.errors import AprilError
from services.april_runtime.backend import RuntimeBackend
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.prompt_templates import render_prompt
from services.april_runtime.schemas import ChatMessage


@dataclass(frozen=True, slots=True)
class ContextResult:
    messages: list[ChatMessage]
    truncated: bool
    input_tokens: int
    reserved_output_tokens: int
    removed_message_count: int
    truncated_tool_result_count: int
    selected_context_limit: int

    def metadata(self) -> dict[str, int | bool]:
        return {
            "estimated_input_tokens": self.input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "removed_message_count": self.removed_message_count,
            "truncated_tool_result_count": self.truncated_tool_result_count,
            "selected_context_limit": self.selected_context_limit,
            "truncated": self.truncated,
        }


class ContextManager:
    async def fit(
        self,
        *,
        model: ModelDefinition,
        backend: RuntimeBackend,
        messages: list[ChatMessage],
        max_output_tokens: int,
    ) -> ContextResult:
        budget = model.context_size - max_output_tokens
        if budget <= 0:
            raise AprilError(
                "CONTEXT_BUDGET_EXCEEDED",
                "Model context window is too small after reserving output tokens.",
                400,
                {"context_size": model.context_size, "reserved_output_tokens": max_output_tokens},
            )

        system_indexes = {
            index for index, message in enumerate(messages) if message.role == "system"
        }
        latest_user_index = _latest_user_index(messages)
        required_indexes = set(system_indexes)
        if latest_user_index is not None:
            required_indexes.add(latest_user_index)

        selected_indexes = set(required_indexes)
        selected_messages = _messages_for_indexes(messages, selected_indexes)
        total = await self._count_rendered_tokens(model, backend, selected_messages)
        if total > budget:
            raise AprilError(
                "CONTEXT_BUDGET_EXCEEDED",
                "Required system prompt and latest request exceed the model context budget.",
                400,
                {
                    "estimated_input_tokens": total,
                    "selected_context_limit": budget,
                    "reserved_output_tokens": max_output_tokens,
                },
            )

        removed = 0
        truncated_tools = 0
        for index in range(len(messages) - 1, -1, -1):
            if index in selected_indexes:
                continue
            candidate_indexes = {*selected_indexes, index}
            candidate_messages = _messages_for_indexes(messages, candidate_indexes)
            candidate_total = await self._count_rendered_tokens(model, backend, candidate_messages)
            if candidate_total <= budget:
                selected_indexes.add(index)
                total = candidate_total
                continue
            message = messages[index]
            if message.role == "tool":
                truncated = await self._fit_truncated_tool(
                    model=model,
                    backend=backend,
                    messages=messages,
                    selected_indexes=selected_indexes,
                    tool_index=index,
                    budget=budget,
                )
                if truncated is not None:
                    messages = truncated.messages
                    selected_indexes.add(index)
                    total = truncated.input_tokens
                    truncated_tools += 1
                    continue
            removed += 1

        selected_messages = _messages_for_indexes(messages, selected_indexes)
        total = await self._count_rendered_tokens(model, backend, selected_messages)
        return ContextResult(
            messages=selected_messages,
            truncated=removed > 0 or truncated_tools > 0,
            input_tokens=total,
            reserved_output_tokens=max_output_tokens,
            removed_message_count=removed,
            truncated_tool_result_count=truncated_tools,
            selected_context_limit=budget,
        )

    async def _count_rendered_tokens(
        self,
        model: ModelDefinition,
        backend: RuntimeBackend,
        messages: list[ChatMessage],
    ) -> int:
        return await backend.count_tokens(render_prompt(model, messages))

    async def _fit_truncated_tool(
        self,
        *,
        model: ModelDefinition,
        backend: RuntimeBackend,
        messages: list[ChatMessage],
        selected_indexes: set[int],
        tool_index: int,
        budget: int,
    ) -> ContextResult | None:
        original = messages[tool_index].content
        marker = "\n[TRUNCATED]"
        low = 0
        high = len(original)
        best_messages: list[ChatMessage] | None = None
        best_total = 0
        while low <= high:
            midpoint = (low + high) // 2
            candidate_content = original[:midpoint].rstrip() + marker
            candidate_all = list(messages)
            candidate_all[tool_index] = messages[tool_index].model_copy(
                update={"content": candidate_content}
            )
            candidate = _messages_for_indexes(candidate_all, {*selected_indexes, tool_index})
            total = await self._count_rendered_tokens(model, backend, candidate)
            if total <= budget:
                best_messages = candidate_all
                best_total = total
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best_messages is None:
            return None
        return ContextResult(
            messages=best_messages,
            truncated=True,
            input_tokens=best_total,
            reserved_output_tokens=0,
            removed_message_count=0,
            truncated_tool_result_count=1,
            selected_context_limit=budget,
        )


def _latest_user_index(messages: list[ChatMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return None


def _messages_for_indexes(messages: list[ChatMessage], indexes: set[int]) -> list[ChatMessage]:
    return [message for index, message in enumerate(messages) if index in indexes]
