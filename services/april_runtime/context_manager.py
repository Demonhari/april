from __future__ import annotations

import json
from dataclasses import dataclass

from april_common.errors import AprilError
from services.april_runtime.backend import RuntimeBackend
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.prompt_templates import render_prompt
from services.april_runtime.schemas import ChatMessage

SUMMARY_BLOCK_PREFIX = "[MACHINE-GENERATED CONVERSATION CONTEXT"
TRUNCATION_MARKER = "[TRUNCATED]"


@dataclass(frozen=True, slots=True)
class ContextGroup:
    messages: tuple[ChatMessage, ...]
    kind: str
    required: bool = False
    complete: bool = True
    orphan_tool_count: int = 0

    @property
    def has_tool_result(self) -> bool:
        return any(message.role == "tool" for message in self.messages)


@dataclass(frozen=True, slots=True)
class ContextResult:
    messages: list[ChatMessage]
    truncated: bool
    input_tokens: int
    reserved_output_tokens: int
    removed_message_count: int
    truncated_tool_result_count: int
    selected_context_limit: int
    removed_group_count: int = 0
    orphan_tool_message_count: int = 0
    complete_group_count: int = 0
    conversation_summary_included: bool = False
    context_continuity: str = "message_window_only"
    context_warning_codes: tuple[str, ...] = ()

    def metadata(self) -> dict[str, object]:
        return {
            "estimated_input_tokens": self.input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "removed_message_count": self.removed_message_count,
            "removed_group_count": self.removed_group_count,
            "truncated_tool_result_count": self.truncated_tool_result_count,
            "orphan_tool_message_count": self.orphan_tool_message_count,
            "complete_group_count": self.complete_group_count,
            "conversation_summary_included": self.conversation_summary_included,
            "context_continuity": self.context_continuity,
            "context_warning_codes": list(self.context_warning_codes),
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
        metadata: dict[str, object] | None = None,
    ) -> ContextResult:
        budget = model.context_size - max_output_tokens
        if budget <= 0:
            raise AprilError(
                "CONTEXT_BUDGET_EXCEEDED",
                "Model context window is too small after reserving output tokens.",
                400,
                {"context_size": model.context_size, "reserved_output_tokens": max_output_tokens},
            )

        groups = build_context_groups(messages)
        orphan_tools = sum(group.orphan_tool_count for group in groups)
        complete_groups = sum(group.complete and group.kind != "orphan" for group in groups)
        selectable = [
            group
            for group in groups
            if group.kind != "orphan"
            and (
                group.required or group.complete or group.kind in {"system", "conversation_summary"}
            )
        ]
        required = [group for group in selectable if group.required]
        summary_groups = [group for group in selectable if group.kind == "conversation_summary"]

        selected: dict[int, ContextGroup] = {id(group): group for group in required}
        selected_messages = _flatten_selected(groups, selected)
        total = await self._count_rendered_tokens(model, backend, selected_messages, metadata)
        truncated_tools = 0
        if total > budget:
            for group in required:
                if not group.has_tool_result:
                    continue
                without_group = {key: value for key, value in selected.items() if key != id(group)}
                fitted = await self._fit_truncated_tool_group(
                    model=model,
                    backend=backend,
                    all_groups=groups,
                    selected=without_group,
                    group=group,
                    budget=budget,
                    metadata=metadata,
                )
                if fitted is not None:
                    fitted_group, total, count = fitted
                    selected[id(group)] = fitted_group
                    truncated_tools += count
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

        for group in summary_groups:
            if id(group) in selected:
                continue
            candidate = {**selected, id(group): group}
            candidate_messages = _flatten_selected(groups, candidate)
            candidate_total = await self._count_rendered_tokens(
                model, backend, candidate_messages, metadata
            )
            if candidate_total <= budget:
                selected = candidate
                total = candidate_total
        newest_first = [
            group
            for group in reversed(selectable)
            if not group.required and group.kind != "conversation_summary"
        ]
        for group in newest_first:
            candidate = {**selected, id(group): group}
            candidate_messages = _flatten_selected(groups, candidate)
            candidate_total = await self._count_rendered_tokens(
                model, backend, candidate_messages, metadata
            )
            if candidate_total <= budget:
                selected = candidate
                total = candidate_total
                continue
            if not group.has_tool_result:
                break
            fitted = await self._fit_truncated_tool_group(
                model=model,
                backend=backend,
                all_groups=groups,
                selected=selected,
                group=group,
                budget=budget,
                metadata=metadata,
            )
            if fitted is not None:
                fitted_group, total, count = fitted
                selected[id(group)] = fitted_group
                truncated_tools += count
                continue
            break

        selected_messages = _flatten_selected(groups, selected)
        total = await self._count_rendered_tokens(model, backend, selected_messages, metadata)
        selected_original_message_count = sum(
            len(group.messages) for group in groups if id(group) in selected
        )
        removed_messages = len(messages) - selected_original_message_count
        removed_groups = sum(id(group) not in selected for group in groups)
        summary_included = any(
            group.kind == "conversation_summary" and id(group) in selected for group in groups
        )
        warning_codes = (
            ("context_truncated_without_persisted_summary",)
            if removed_messages > 0 and not summary_included
            else ()
        )
        return ContextResult(
            messages=selected_messages,
            truncated=removed_messages > 0 or truncated_tools > 0,
            input_tokens=total,
            reserved_output_tokens=max_output_tokens,
            removed_message_count=removed_messages,
            removed_group_count=removed_groups,
            truncated_tool_result_count=truncated_tools,
            orphan_tool_message_count=orphan_tools,
            complete_group_count=complete_groups,
            conversation_summary_included=summary_included,
            context_continuity=(
                "summary_plus_recent" if summary_included else "message_window_only"
            ),
            context_warning_codes=warning_codes,
            selected_context_limit=budget,
        )

    async def _count_rendered_tokens(
        self,
        model: ModelDefinition,
        backend: RuntimeBackend,
        messages: list[ChatMessage],
        metadata: dict[str, object] | None = None,
    ) -> int:
        return await backend.count_tokens(render_prompt(model, messages, metadata=metadata))

    async def _fit_truncated_tool_group(
        self,
        *,
        model: ModelDefinition,
        backend: RuntimeBackend,
        all_groups: list[ContextGroup],
        selected: dict[int, ContextGroup],
        group: ContextGroup,
        budget: int,
        metadata: dict[str, object] | None,
    ) -> tuple[ContextGroup, int, int] | None:
        tool_indexes = [
            index for index, message in enumerate(group.messages) if message.role == "tool"
        ]
        if not tool_indexes:
            return None

        originals = {index: group.messages[index].content for index in tool_indexes}
        low = 0
        high = max(len(value) for value in originals.values())
        best: tuple[ContextGroup, int] | None = None
        while low <= high:
            midpoint = (low + high) // 2
            candidate_messages = list(group.messages)
            for index in tool_indexes:
                candidate_messages[index] = candidate_messages[index].model_copy(
                    update={
                        "content": _truncate_tool_result(
                            originals[index], max_content_chars=midpoint
                        )
                    }
                )
            candidate_group = ContextGroup(
                messages=tuple(candidate_messages),
                kind=group.kind,
                required=group.required,
                complete=group.complete,
            )
            candidate_selected = {**selected, id(group): candidate_group}
            flattened = _flatten_selected(all_groups, candidate_selected)
            total = await self._count_rendered_tokens(model, backend, flattened, metadata)
            if total <= budget:
                best = candidate_group, total
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best is None:
            return None
        return best[0], best[1], len(tool_indexes)


def build_context_groups(messages: list[ChatMessage]) -> list[ContextGroup]:
    """Build deterministic, ordered groups and isolate orphan tool messages."""

    groups: list[ContextGroup] = []
    latest_user_index = _latest_user_index(messages)
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "system":
            is_summary = message.content.startswith(SUMMARY_BLOCK_PREFIX)
            groups.append(
                ContextGroup(
                    messages=(message,),
                    kind="conversation_summary" if is_summary else "system",
                    required=not is_summary,
                )
            )
            index += 1
            continue
        if message.role == "user":
            end = index + 1
            while end < len(messages) and messages[end].role not in {"user", "system"}:
                end += 1
            turn_messages, orphan_groups = _sanitize_turn(messages[index:end])
            groups.append(
                ContextGroup(
                    messages=tuple(turn_messages),
                    kind="conversation",
                    required=index == latest_user_index,
                    complete=_turn_complete(turn_messages),
                )
            )
            groups.extend(orphan_groups)
            index = end
            continue
        if message.role == "assistant" and _is_tool_request(message.content):
            end = index + 1
            saw_tool = False
            while end < len(messages):
                candidate = messages[end]
                if candidate.role == "tool":
                    saw_tool = True
                    end += 1
                    continue
                if saw_tool and candidate.role == "assistant":
                    end += 1
                break
            sequence = tuple(messages[index:end])
            groups.append(
                ContextGroup(
                    messages=sequence,
                    kind="tool_sequence",
                    complete=(
                        saw_tool
                        and sequence[-1].role == "assistant"
                        and not _is_tool_request(sequence[-1].content)
                    ),
                )
            )
            index = end
            continue
        warning_count = 1 if message.role == "tool" else 0
        groups.append(
            ContextGroup(
                messages=(message,),
                kind="orphan",
                complete=False,
                orphan_tool_count=warning_count,
            )
        )
        index += 1
    return groups


def _sanitize_turn(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], list[ContextGroup]]:
    selected: list[ChatMessage] = []
    orphans: list[ContextGroup] = []
    tool_request_open = False
    tool_seen = False
    for message in messages:
        if message.role == "assistant":
            if tool_request_open:
                if tool_seen:
                    tool_request_open = False
                    tool_seen = False
                elif not _is_tool_request(message.content):
                    # A continuation without the requested tool result is
                    # detached and cannot be retained, even in the latest turn.
                    orphans.append(
                        ContextGroup(
                            messages=(message,),
                            kind="orphan",
                            complete=False,
                        )
                    )
                    tool_request_open = False
                    continue
            if _is_tool_request(message.content):
                tool_request_open = True
                tool_seen = False
            selected.append(message)
        elif message.role == "tool":
            if tool_request_open:
                selected.append(message)
                tool_seen = True
            else:
                orphans.append(
                    ContextGroup(
                        messages=(message,),
                        kind="orphan",
                        complete=False,
                        orphan_tool_count=1,
                    )
                )
        else:
            selected.append(message)
    return selected, orphans


def _turn_complete(messages: list[ChatMessage]) -> bool:
    if not messages or messages[0].role != "user":
        return False
    if messages[-1].role != "assistant":
        return False
    return not _is_tool_request(messages[-1].content)


def _is_tool_request(content: str) -> bool:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(value, dict) and value.get("type") == "tool_request"


def _truncate_tool_result(content: str, *, max_content_chars: int) -> str:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        value = {}
    tool = value.get("tool", "unknown") if isinstance(value, dict) else "unknown"
    ok = value.get("ok") if isinstance(value, dict) else None
    prefix = content[:max_content_chars].rstrip()
    payload = {
        "tool": tool,
        "ok": ok,
        "output": f"{prefix}\n{TRUNCATION_MARKER}" if prefix else TRUNCATION_MARKER,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _flatten_selected(
    groups: list[ContextGroup], selected: dict[int, ContextGroup]
) -> list[ChatMessage]:
    flattened: list[ChatMessage] = []
    for group in groups:
        chosen = selected.get(id(group))
        if chosen is not None:
            flattened.extend(chosen.messages)
    return flattened


def _latest_user_index(messages: list[ChatMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return None
