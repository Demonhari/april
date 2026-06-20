from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from pydantic import ValidationError as PydanticValidationError

from april_common.errors import ValidationError
from services.brain.schemas import BrainDecision

RepairCallback = Callable[[str], Awaitable[str]]


def extract_single_json_object(text: str) -> str:
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


def parse_brain_decision(text: str, *, method: str = "model") -> BrainDecision:
    raw = extract_single_json_object(text)
    try:
        data = json.loads(raw)
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
