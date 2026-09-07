from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, Field

from apps.runner.mac_report import RoutingReport, routing_report_from_results
from services.brain.deterministic_router import DeterministicRouter
from services.brain.fallback_router import FallbackRouter
from services.brain.schemas import BrainDecision
from services.memory.database import connect_sqlite

from .verify import RealModelVerifier


class BrainEvalCase(BaseModel):
    id: str
    message: str
    expected_intent: str
    expected_agent: str
    expected_model_id: str | None = None
    expected_tools: list[str] | None = None
    expected_permission_level: int | None = None
    expected_risk_level: str | None = None
    expected_needs_confirmation: bool | None = None
    expected_routing_method: str | None = None


class BrainEvalResult(BaseModel):
    id: str
    ok: bool
    schema_valid: bool = True
    routing_ok: bool = True
    expected_intent: str
    expected_agent: str
    expected_model_id: str | None = None
    expected_tools: list[str] | None = None
    expected_permission_level: int | None = None
    expected_risk_level: str | None = None
    expected_needs_confirmation: bool | None = None
    actual: dict[str, Any] = Field(default_factory=dict)
    route_source: str | None = None
    routing_failure_code: str | None = None
    fallback_reason: str | None = None
    detail: str = ""
    mismatch_codes: list[str] = Field(default_factory=list)
    stage_code: str | None = None
    downstream_ok: bool | None = None
    repair_attempted: bool = False
    repair_succeeded: bool = False
    finish_reason: str | None = None
    context_truncated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    proposal_operation: str | None = None
    proposal_context: str | None = None
    contract_fingerprint: str | None = None
    first_proposal_operation: str | None = None
    first_proposal_context: str | None = None
    first_proposal_tool_class: str | None = None
    first_rejection_code: str | None = None
    repair_proposal_operation: str | None = None
    repair_rejection_code: str | None = None
    coercions: list[str] = Field(default_factory=list)
    downstream_status: int | None = None
    downstream_error_code: str | None = None
    downstream_runtime_error_code: str | None = None


def load_brain_eval_cases(home: Path) -> list[BrainEvalCase]:
    path = home / "tests" / "fixtures" / "evals" / "brain_routes.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("brain eval fixture cases must be a list")
    return [BrainEvalCase.model_validate(item) for item in cases]


def run_fake_brain_eval(home: Path) -> list[BrainEvalResult]:
    deterministic = DeterministicRouter()
    router = FallbackRouter()
    results: list[BrainEvalResult] = []
    for case in load_brain_eval_cases(home):
        match = deterministic.route(case.message)
        if match is not None:
            actual = match.decision.model_dump()
            actual.update(
                routing_method="deterministic",
                route_source="deterministic",
                route_provenance="trusted_v1",
            )
        else:
            actual = router.route(case.message).model_dump()
        results.append(_evaluate_case(case, actual, schema_valid=True))
    return results


def _evaluate_case(
    case: BrainEvalCase,
    actual: dict[str, Any],
    *,
    schema_valid: bool,
    allow_fallback: bool = True,
    evidence: dict[str, Any] | None = None,
) -> BrainEvalResult:
    mismatches: list[str] = []
    _expect(mismatches, "intent", case.expected_intent, actual.get("intent"))
    _expect(mismatches, "agent", case.expected_agent, actual.get("agent"))
    _expect_optional(mismatches, "model_id", case.expected_model_id, actual.get("model_id"))
    if case.expected_tools is not None:
        actual_tools = actual.get("tools_needed", [])
        if sorted(actual_tools) != sorted(case.expected_tools):
            mismatches.append(f"tools expected {case.expected_tools!r}, got {actual_tools!r}")
    _expect_optional(
        mismatches,
        "permission_level",
        case.expected_permission_level,
        actual.get("permission_level"),
    )
    _expect_optional(mismatches, "risk_level", case.expected_risk_level, actual.get("risk_level"))
    _expect_optional(
        mismatches,
        "needs_confirmation",
        case.expected_needs_confirmation,
        actual.get("needs_confirmation"),
    )
    actual_method = _evaluation_route_source(actual, trusted_only=not allow_fallback)
    if allow_fallback:
        # Fake/fallback eval: the fixture's expected routing_method (e.g. fallback)
        # is authoritative.
        _expect_optional(mismatches, "routing_method", case.expected_routing_method, actual_method)
    else:
        # Real-model eval: a fallback route means the model JSON was unusable (or the
        # runtime failed) and the deterministic fallback router answered instead — a
        # failure. Only a real model/model-repair route is acceptable.
        if actual_method == "fallback":
            mismatches.append("routing_method was fallback (model JSON unusable or runtime failed)")
        elif actual_method not in {"deterministic", "model", "model_repair"}:
            mismatches.append(f"trusted routing provenance is unknown: {actual_method!r}")
    evidence = evidence or {}
    downstream_ok = evidence.get("downstream_ok")
    if downstream_ok is False:
        mismatches.append("downstream_chat_failure")
    routing_ok = not mismatches or mismatches == ["downstream_chat_failure"]
    mismatch_codes = _mismatch_codes(
        case,
        actual,
        schema_valid=schema_valid,
        allow_fallback=allow_fallback,
        trusted_only=not allow_fallback,
        evidence=evidence,
    )
    if downstream_ok is False and "downstream_chat_failure" not in mismatch_codes:
        mismatch_codes.append("downstream_chat_failure")
    return BrainEvalResult(
        id=case.id,
        ok=schema_valid and routing_ok and downstream_ok is not False,
        schema_valid=schema_valid,
        routing_ok=routing_ok,
        expected_intent=case.expected_intent,
        expected_agent=case.expected_agent,
        expected_model_id=case.expected_model_id,
        expected_tools=case.expected_tools,
        expected_permission_level=case.expected_permission_level,
        expected_risk_level=case.expected_risk_level,
        expected_needs_confirmation=case.expected_needs_confirmation,
        actual=actual,
        route_source=(actual_method),
        routing_failure_code=(
            str(evidence["routing_failure_code"])
            if isinstance(evidence.get("routing_failure_code"), str)
            else None
        ),
        fallback_reason=(
            str(evidence["fallback_reason"])
            if isinstance(evidence.get("fallback_reason"), str)
            else None
        ),
        detail=(
            ""
            if schema_valid and routing_ok and downstream_ok is not False
            else "; ".join(mismatches or ["schema invalid"])
        ),
        mismatch_codes=mismatch_codes,
        stage_code=evidence.get("stage_code"),
        downstream_ok=downstream_ok if isinstance(downstream_ok, bool) else None,
        repair_attempted=bool(evidence.get("repair_attempted", False)),
        repair_succeeded=bool(evidence.get("repair_succeeded", False)),
        finish_reason=(
            str(evidence["finish_reason"])
            if isinstance(evidence.get("finish_reason"), str)
            else None
        ),
        context_truncated=bool(evidence.get("context_truncated", False)),
        input_tokens=int(evidence.get("input_tokens", 0) or 0),
        output_tokens=int(evidence.get("output_tokens", 0) or 0),
        proposal_operation=(
            str(evidence["proposal_operation"])
            if isinstance(evidence.get("proposal_operation"), str)
            else None
        ),
        proposal_context=(
            str(evidence["proposal_context"])
            if isinstance(evidence.get("proposal_context"), str)
            else None
        ),
        contract_fingerprint=(
            str(evidence["contract_fingerprint"])
            if isinstance(evidence.get("contract_fingerprint"), str)
            else None
        ),
        first_proposal_operation=_bounded_evidence_text(evidence.get("first_proposal_operation")),
        first_proposal_context=_bounded_evidence_text(evidence.get("first_proposal_context")),
        first_proposal_tool_class=_bounded_evidence_text(evidence.get("first_proposal_tool_class")),
        first_rejection_code=_bounded_evidence_text(evidence.get("first_rejection_code")),
        repair_proposal_operation=_bounded_evidence_text(evidence.get("repair_proposal_operation")),
        repair_rejection_code=_bounded_evidence_text(evidence.get("repair_rejection_code")),
        coercions=[item[:64] for item in evidence.get("coercions", []) if isinstance(item, str)][
            :8
        ],
        downstream_status=(
            int(evidence["downstream_status"])
            if isinstance(evidence.get("downstream_status"), int)
            else None
        ),
        downstream_error_code=_bounded_evidence_text(evidence.get("downstream_error_code")),
        downstream_runtime_error_code=_bounded_evidence_text(
            evidence.get("downstream_runtime_error_code")
        ),
    )


def _mismatch_codes(
    case: BrainEvalCase,
    actual: dict[str, Any],
    *,
    schema_valid: bool,
    allow_fallback: bool,
    trusted_only: bool = False,
    evidence: dict[str, Any] | None = None,
) -> list[str]:
    codes: list[str] = []
    evidence = evidence or {}
    if not actual:
        stage_code = evidence.get("stage_code")
        return [stage_code] if isinstance(stage_code, str) and stage_code else []
    if not schema_valid:
        codes.append("schema_invalid")
        return codes
    for key, code in (
        ("intent", "intent_mismatch"),
        ("agent", "agent_mismatch"),
        ("model_id", "model_role_mismatch"),
        ("permission_level", "permission_mismatch"),
        ("risk_level", "risk_mismatch"),
        ("needs_confirmation", "confirmation_mismatch"),
    ):
        expected = getattr(case, f"expected_{key}", None)
        if expected is not None and actual.get(key) != expected:
            codes.append(code)
    if case.expected_tools is not None and sorted(actual.get("tools_needed", [])) != sorted(
        case.expected_tools
    ):
        codes.append("tool_set_mismatch")
    method = _evaluation_route_source(actual, trusted_only=trusted_only)
    if allow_fallback:
        if case.expected_routing_method is not None and method != case.expected_routing_method:
            codes.append("routing_method_mismatch")
    elif method not in {"deterministic", "model", "model_repair"}:
        codes.append("fallback_route" if method == "fallback" else "routing_provenance_invalid")
    return codes


def _bounded_evidence_text(value: object) -> str | None:
    return value[:64] if isinstance(value, str) and value else None


def real_routing_report(
    cases: list[BrainEvalCase],
    decisions: list[Any],
    evidence: list[dict[str, Any]] | None = None,
) -> RoutingReport:
    """Build a real-mode RoutingReport, disallowing fallback for every case.

    ``decisions[i]`` is the Brain decision recorded for ``cases[i]`` (or an empty
    dict when the request errored). Used by the all-configured-models verifier so
    its routing report counts a schema-valid fallback decision as a failure.
    """
    results: list[BrainEvalResult] = []
    for index, case in enumerate(cases):
        actual = decisions[index] if index < len(decisions) else {}
        actual_dict, schema_valid = _validated_decision(actual)
        results.append(
            _evaluate_case(
                case,
                actual_dict,
                schema_valid=schema_valid,
                allow_fallback=False,
                evidence=(evidence[index] if evidence and index < len(evidence) else None),
            )
        )
    return routing_report_from_results(
        results,
        require_trusted_provenance=True,
        expected_case_ids=[case.id for case in cases],
    )


def _validated_decision(value: Any) -> tuple[dict[str, Any], bool]:
    if not value:
        return {}, False
    try:
        decision = BrainDecision.model_validate(value)
    except ValueError:
        # Preserve the fact that a non-empty response existed without retaining
        # generated text in the report.  Empty actual evidence is reserved for
        # transport/runtime failures and missing correlated events.
        return value if isinstance(value, dict) else {"_invalid_response": True}, False
    normalized = decision.model_dump()
    # route_source is trusted only when it came from the redacted, persisted
    # orchestrator event. A model cannot manufacture this marker in its own JSON.
    if isinstance(value, dict) and value.get("route_provenance") in {
        "trusted_v1",
        "trusted_model_only_v1",
    }:
        source = value.get("route_source")
        if source in {"deterministic", "model", "model_repair", "fallback"}:
            normalized["route_source"] = source
            normalized["route_provenance"] = "trusted_v1"
    return normalized, True


def _evaluation_route_source(actual: dict[str, Any], *, trusted_only: bool = False) -> str | None:
    source = actual.get("route_source")
    if actual.get("route_provenance") in {"trusted_v1", "trusted_model_only_v1"} and source in {
        "deterministic",
        "model",
        "model_repair",
        "fallback",
    }:
        return str(source)
    # Compatibility for in-memory/unit callers and pre-provenance decisions.
    # Persisted reports without the trusted marker are handled as unknown by the
    # report reader rather than being upgraded to a real-model claim.
    if trusted_only:
        return None
    method = actual.get("routing_method")
    return str(method) if method in {"model", "model_repair", "fallback"} else None


def _expect(mismatches: list[str], key: str, expected: object, actual: object) -> None:
    if actual != expected:
        mismatches.append(f"{key} expected {expected!r}, got {actual!r}")


def _expect_optional(
    mismatches: list[str], key: str, expected: object | None, actual: object
) -> None:
    if expected is not None:
        _expect(mismatches, key, expected, actual)


class RealBrainEvalRunner(
    RealModelVerifier
):  # pragma: no cover - requires optional real GGUF runtime
    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def run_eval(self) -> list[BrainEvalResult]:
        results: list[BrainEvalResult] = []
        cases = load_brain_eval_cases(self.repo_home)
        try:
            self._prepare()
            env = self._env()
            self.runtime = self._start("services.april_runtime.server", env, self.runtime_log)
            self.api = self._start("services.api.server", env, self.api_log)
            self._wait_json(self.runtime_url + "/runtime/health", auth_runtime=True)
            self._wait_json(self.api_url + "/health")
            with httpx.Client(
                base_url=self.api_url,
                headers=self.headers,
                timeout=self.timeout,
            ) as client:
                for case in cases:
                    results.append(self._run_case(client, case))
        finally:
            self._stop()
            shutil.rmtree(self.temp, ignore_errors=True)
        return results

    def _run_case(self, client: httpx.Client, case: BrainEvalCase) -> BrainEvalResult:
        marker = self._brain_decision_marker()
        response = client.post("/chat", json={"message": case.message})
        if response.status_code >= 400:
            return BrainEvalResult(
                id=case.id,
                ok=False,
                schema_valid=False,
                routing_ok=False,
                expected_intent=case.expected_intent,
                expected_agent=case.expected_agent,
                detail=response.text[:500],
            )
        actual, schema_valid = _validated_decision(self._brain_decision_after(marker))
        # Real-model eval: fallback routing is a failure, not an accepted route.
        return _evaluate_case(case, actual, schema_valid=schema_valid, allow_fallback=False)

    def _latest_decision(self) -> dict[str, Any]:
        database = self.temp / "data" / "april.db"
        with connect_sqlite(database) as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM conversation_events
                WHERE event_type = 'brain_decision'
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(str(row[0]))
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}


def run_real_brain_eval(home: Path, model_path: Path) -> list[BrainEvalResult]:
    return RealBrainEvalRunner(home=home, model_path=model_path).run_eval()
