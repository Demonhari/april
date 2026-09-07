from __future__ import annotations

from typing import Any

import httpx

from services.brain.model_routing import ModelRoutingOutcome


def model_route_evidence(outcome: ModelRoutingOutcome, fingerprint: str) -> dict[str, Any]:
    return {
        "stage_code": outcome.failure_code,
        "repair_attempted": outcome.repair_attempted,
        "repair_succeeded": outcome.repair_succeeded,
        "finish_reason": outcome.finish_reason,
        "context_truncated": outcome.context_truncated,
        "input_tokens": outcome.input_tokens,
        "output_tokens": outcome.output_tokens,
        "proposal_operation": outcome.first_proposal_operation,
        "proposal_context": outcome.first_proposal_context,
        "first_proposal_operation": outcome.first_proposal_operation,
        "first_proposal_context": outcome.first_proposal_context,
        "first_proposal_tool_class": outcome.first_proposal_tool_class,
        "first_rejection_code": outcome.first_rejection_code,
        "repair_proposal_operation": outcome.repair_proposal_operation,
        "repair_rejection_code": outcome.repair_rejection_code,
        "coercions": outcome.coercions,
        "contract_fingerprint": fingerprint,
    }


def persisted_route_evidence(event: dict[str, Any], response: httpx.Response) -> dict[str, Any]:
    return {
        "stage_code": _routing_stage_code(event, response),
        "downstream_ok": response.status_code < 400,
        "downstream_status": response.status_code,
        "downstream_error_code": _downstream_error_code(response),
        "routing_failure_code": event.get("routing_failure_code") if event else None,
        "fallback_reason": event.get("fallback_reason") if event else None,
        "proposal_operation": event.get("proposal_operation"),
        "proposal_context": event.get("proposal_context"),
        "first_proposal_operation": event.get("first_proposal_operation"),
        "first_proposal_context": event.get("first_proposal_context"),
        "first_proposal_tool_class": event.get("first_proposal_tool_class"),
        "first_rejection_code": event.get("first_rejection_code"),
        "repair_proposal_operation": event.get("repair_proposal_operation"),
        "repair_rejection_code": event.get("repair_rejection_code"),
        "coercions": event.get("coercions", []),
        "repair_attempted": event.get("repair_attempted", False),
        "repair_succeeded": event.get("repair_succeeded", False),
        "contract_fingerprint": event.get("contract_fingerprint"),
    }


def runtime_unavailable_evidence(
    count: int, fingerprint: str | None = None
) -> list[dict[str, Any]]:
    return [
        {"stage_code": "runtime_unavailable", "contract_fingerprint": fingerprint}
        for _ in range(count)
    ]


def _downstream_error_code(response: httpx.Response) -> str | None:
    if response.status_code < 400:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    code = payload["error"].get("code")
    return code[:64] if isinstance(code, str) and code else None


def _routing_stage_code(event: dict[str, Any], response: httpx.Response) -> str | None:
    if event and response.status_code < 400:
        return None
    failure_code = event.get("routing_failure_code") if event else None
    route_source = (event.get("route_source") or event.get("routing_method")) if event else None
    if route_source == "fallback" and failure_code in {
        "runtime_unavailable",
        "inference_transport_error",
    }:
        return "runtime_unavailable"
    body = response.text[:500].lower()
    if response.status_code in {502, 503} and "runtime" in body:
        return "runtime_unavailable"
    return "downstream_chat_failure" if event else "missing_correlated_route_event"
