from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, Field

from services.brain.fallback_router import FallbackRouter

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
    actual: dict[str, Any] = Field(default_factory=dict)
    detail: str = ""


def load_brain_eval_cases(home: Path) -> list[BrainEvalCase]:
    path = home / "tests" / "fixtures" / "evals" / "brain_routes.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("brain eval fixture cases must be a list")
    return [BrainEvalCase.model_validate(item) for item in cases]


def run_fake_brain_eval(home: Path) -> list[BrainEvalResult]:
    router = FallbackRouter()
    results: list[BrainEvalResult] = []
    for case in load_brain_eval_cases(home):
        decision = router.route(case.message)
        actual = decision.model_dump()
        results.append(_evaluate_case(case, actual, schema_valid=True))
    return results


def _evaluate_case(
    case: BrainEvalCase,
    actual: dict[str, Any],
    *,
    schema_valid: bool,
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
    _expect_optional(
        mismatches,
        "routing_method",
        case.expected_routing_method,
        actual.get("routing_method"),
    )
    routing_ok = not mismatches
    return BrainEvalResult(
        id=case.id,
        ok=schema_valid and routing_ok,
        schema_valid=schema_valid,
        routing_ok=routing_ok,
        expected_intent=case.expected_intent,
        expected_agent=case.expected_agent,
        actual=actual,
        detail="" if schema_valid and routing_ok else "; ".join(mismatches),
    )


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
        actual = self._latest_decision()
        try:
            from services.brain.schemas import BrainDecision

            BrainDecision.model_validate(actual)
            schema_valid = True
        except ValueError:
            schema_valid = False
        return _evaluate_case(case, actual, schema_valid=schema_valid)

    def _latest_decision(self) -> dict[str, Any]:
        database = self.temp / "data" / "april.db"
        with sqlite3.connect(database) as conn:
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
