from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

from pydantic import ValidationError as PydanticValidationError

from april_common.errors import ValidationError
from services.brain.schemas import BrainDecision

RepairCallback = Callable[[str], Awaitable[str]]


def extract_single_json_object(text: str) -> str:
    text = _strip_markdown_fence(text)
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
            if depth < 0:
                raise ValidationError("Malformed JSON object.")
    if len(objects) != 1:
        raise ValidationError("Expected exactly one JSON object.", {"count": len(objects)})
    return objects[0]


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _with_missing_optional_arrays(data: dict[str, object]) -> dict[str, object]:
    for key in ("tools_needed", "planned_tool_calls", "memory_queries", "task_steps"):
        data.setdefault(key, [])
    return data


def parse_brain_decision(text: str, *, method: str = "model") -> BrainDecision:
    raw = _remove_trailing_commas(extract_single_json_object(text))
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = _with_missing_optional_arrays(data)
        decision = BrainDecision.model_validate(data)
    except (json.JSONDecodeError, PydanticValidationError) as exc:
        raise ValidationError(
            "Brain JSON did not match the routing schema.", {"error": str(exc)}
        ) from exc
    return decision.model_copy(update={"routing_method": method})


async def parse_with_repair(text: str, repair: RepairCallback) -> BrainDecision:
    try:
        return parse_brain_decision(text, method="model")
    except ValidationError:
        repaired = await repair(text)
        return parse_brain_decision(repaired, method="model_repair")
