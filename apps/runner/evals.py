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


class BrainEvalResult(BaseModel):
    id: str
    ok: bool
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
        ok = decision.intent == case.expected_intent and decision.agent == case.expected_agent
        results.append(
            BrainEvalResult(
                id=case.id,
                ok=ok,
                expected_intent=case.expected_intent,
                expected_agent=case.expected_agent,
                actual=actual,
                detail="" if ok else "route mismatch",
            )
        )
    return results


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
                expected_intent=case.expected_intent,
                expected_agent=case.expected_agent,
                detail=response.text[:500],
            )
        actual = self._latest_decision()
        ok = (
            actual.get("intent") == case.expected_intent
            and actual.get("agent") == case.expected_agent
        )
        return BrainEvalResult(
            id=case.id,
            ok=ok,
            expected_intent=case.expected_intent,
            expected_agent=case.expected_agent,
            actual=actual,
            detail="" if ok else "route mismatch",
        )

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
