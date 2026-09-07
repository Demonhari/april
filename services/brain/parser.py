from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

from pydantic import ValidationError as PydanticValidationError

from april_common.errors import ValidationError
from services.brain.route_contract import (
    RouteContractError,
    RoutingProposal,
    proposal_from_legacy,
)
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


def _with_default_confidence(data: dict[str, object], *, method: str) -> dict[str, object]:
    if "confidence" not in data:
        data["confidence"] = 0.55 if method == "model_repair" else 0.7
    return data


def parse_brain_decision(text: str, *, method: str = "model") -> BrainDecision:
    raw = _remove_trailing_commas(extract_single_json_object(text))
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = _with_missing_optional_arrays(data)
            data = _with_default_confidence(data, method=method)
        decision = BrainDecision.model_validate(data)
    except (json.JSONDecodeError, PydanticValidationError) as exc:
        raise ValidationError(
            "Brain JSON did not match the routing schema.", {"error": str(exc)}
        ) from exc
    return decision.model_copy(update={"routing_method": method})


def parse_routing_proposal(text: str) -> RoutingProposal:
    """Parse the bounded semantic contract used by the live router.

    ``proposal_from_legacy`` keeps older fake clients and integrations readable;
    it does not preserve generated policy fields or provenance.
    """
    try:
        raw = _remove_trailing_commas(extract_single_json_object(text))
    except ValidationError as exc:
        raise RouteContractError(
            "schema_rejection:no_json_object",
            "Routing proposal did not contain exactly one JSON object.",
        ) from exc
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("routing proposal must be a JSON object")
        proposal = RoutingProposal.model_validate(proposal_from_legacy(data))
    except PydanticValidationError as exc:
        errors = exc.errors()
        location = errors[0]["loc"] if errors else ("unknown",)
        location_text = ".".join(str(part) for part in location) or "unknown"
        raise RouteContractError(
            f"schema_rejection:{location_text}",
            "Routing proposal did not match the semantic contract.",
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise RouteContractError(
            "schema_rejection:no_json_object",
            "Routing proposal did not match the semantic contract.",
        ) from exc
    return proposal


async def parse_with_repair(text: str, repair: RepairCallback) -> BrainDecision:
    try:
        return parse_brain_decision(text, method="model")
    except ValidationError:
        repaired = await repair(text)
        return parse_brain_decision(repaired, method="model_repair")
