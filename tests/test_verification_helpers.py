from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from apps.runner import verify as verify_coordinator
from apps.runner.verification.local_checks import (
    run_local_sandbox_verification,
    run_local_security_integrity_verification,
)
from apps.runner.verification.multi_model import _routing_error_code
from apps.runner.verification.planning import plan_multi_model_verification, skipped_result_for
from apps.runner.verification.reports import (
    brain_decision_after_marker,
    build_workflow_report,
    chat_result_from_response,
    json_object_candidates,
    latest_brain_decision_marker,
    safe_workflow_report_detail,
)
from apps.runner.verification.routing_evidence import (
    _downstream_error_code,
    _routing_stage_code,
    model_route_evidence,
    persisted_route_evidence,
    runtime_unavailable_evidence,
)
from apps.runner.verification.types import VerifyCheck
from apps.runner.verification.workflow import WorkflowVerifier
from apps.runner.verify import TargetMacValidator
from april_common.errors import ConfigError
from services.april_runtime.model_registry import ModelRegistry
from services.brain.model_routing import ModelRoutingOutcome

AllConfiguredModelsVerifier = verify_coordinator.AllConfiguredModelsVerifier


class _ResponseClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = iter(responses)

    def __enter__(self) -> _ResponseClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, *_args: object, **_kwargs: object) -> httpx.Response:
        return next(self.responses)

    def get(self, *_args: object, **_kwargs: object) -> httpx.Response:
        return next(self.responses)


def test_verification_json_helpers_cover_fences_and_malformed_objects() -> None:
    assert json_object_candidates('```json\n{"ok": true,}\n```') == [{"ok": True}]
    assert json_object_candidates('{"bad": }') == []
    assert json_object_candidates('{"a": 1}{"b": 2}') == [{"a": 1}, {"b": 2}]


def test_workflow_report_redacts_sensitive_details(tmp_path: Path) -> None:
    checks = [
        VerifyCheck("ok", True, "ready"),
        VerifyCheck("bad", False, "prompt bearer secret"),
    ]
    report = build_workflow_report(
        checks,
        real_model_requested=True,
        timeout_seconds=2.0,
        max_output_tokens=32,
        config_fingerprint="fingerprint",
    )
    assert report.summary == "fail"
    assert report.real_model_verified is False
    assert report.check_failures == ["bad"]
    assert report.checks[1].detail == "sensitive detail redacted"
    assert safe_workflow_report_detail("decision_summary: private") == "decision_summary redacted"
    path = tmp_path / "workflow.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["summary"] == "fail"


def test_chat_result_and_marker_helpers_fail_closed(tmp_path: Path) -> None:
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://test/chat"),
        json={"result": {"content": "ok"}},
    )
    assert chat_result_from_response(response, context="chat")["content"] == "ok"
    with pytest.raises(RuntimeError, match="missing result"):
        chat_result_from_response(
            httpx.Response(200, request=httpx.Request("POST", "http://test/chat"), text="nope"),
            context="chat",
        )
    missing = tmp_path / "missing.db"
    assert latest_brain_decision_marker(missing) == 0
    assert brain_decision_after_marker(missing, 0) == {}


def test_routing_evidence_codes_are_bounded_and_structural() -> None:
    outcome = ModelRoutingOutcome(
        failure_code="generation_length", coercions=["confidence_defaulted"]
    )
    evidence = model_route_evidence(outcome, "contract")
    assert evidence["stage_code"] == "generation_length"
    assert evidence["coercions"] == ["confidence_defaulted"]
    assert runtime_unavailable_evidence(2, "contract") == [
        {"stage_code": "runtime_unavailable", "contract_fingerprint": "contract"},
        {"stage_code": "runtime_unavailable", "contract_fingerprint": "contract"},
    ]
    response = httpx.Response(
        503,
        request=httpx.Request("POST", "http://test/chat"),
        json={"error": {"code": "RUNTIME_UNAVAILABLE"}},
    )
    assert _downstream_error_code(response) == "RUNTIME_UNAVAILABLE"
    missing_response = httpx.Response(
        500,
        request=httpx.Request("POST", "http://test/chat"),
        json={"error": {"code": "CORE_ERROR"}},
    )
    assert _routing_stage_code({}, missing_response) == "missing_correlated_route_event"
    event = {"route_source": "fallback", "routing_failure_code": "runtime_unavailable"}
    assert _routing_stage_code(event, response) == "runtime_unavailable"
    persisted = persisted_route_evidence(event, response)
    assert persisted["stage_code"] == "runtime_unavailable"


def test_local_checks_report_missing_configuration_without_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "apps.runner.verification.local_checks.load_settings",
        lambda **_kwargs: (_ for _ in ()).throw(ConfigError("invalid")),
    )
    sandbox = run_local_sandbox_verification(tmp_path)
    security = run_local_security_integrity_verification(tmp_path)
    assert sandbox[0].ok is False
    assert security[0].name == "security configuration"


def test_model_planning_redacts_skips_and_non_real_backends(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    registry = ModelRegistry.from_dict(
        {
            "models": {
                "fake": {
                    "id": "fake",
                    "name": "fake",
                    "path": "fake.gguf",
                    "backend": "fake",
                    "role": "brain",
                    "threads": 1,
                    "context_size": 256,
                    "temperature": 0.0,
                    "max_output_tokens": 8,
                }
            }
        },
        root=tmp_path,
    )
    registry_path = tmp_path / "configs" / "models.yaml"
    registry_path.write_text(
        json.dumps({"models": {"fake": registry.get("fake").model_dump(mode="json")}})
    )
    planned = plan_multi_model_verification(tmp_path, llama_available=False)
    assert planned[0].available is False
    result = skipped_result_for(planned[0])
    assert result.available is False
    assert result.path_basename == "fake.gguf"


def test_target_mac_helpers_are_redacted_and_truthful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = TargetMacValidator(
        home=tmp_path,
        model_path=None,
        require_real_model=False,
        max_output_tokens=32,
        timeout=1.0,
    )
    validator._machine_architecture()
    validator._python_version()
    validator._backend_build_info(False)
    validator._model_dependent_skips()
    assert any(check.status == "skip" for check in validator.checks)
    assert validator._check_ok("missing") is False
    monkeypatch.setattr("apps.runner.verification.target_mac.platform.system", lambda: "Linux")
    validator.checks.clear()
    validator._machine_architecture()
    assert validator.checks[0].status == "manual"


def test_multi_model_validation_helpers_are_conservative() -> None:
    verifier = object.__new__(AllConfiguredModelsVerifier)
    assert verifier._valid_coding_plan('{"plan":["edit","test"]}') is True
    assert verifier._valid_coding_plan('{"plan":[1]}') is False
    assert verifier._valid_system_decision('{"execute":false,"permission_level":0}') is True
    assert verifier._valid_system_decision('{"execute":true,"permission_level":0}') is False
    assert (
        verifier._valid_router_decision(
            '{"intent":"planning","agent":"general_agent","model_id":"april-brain",'
            '"confidence":0.7,"permission_level":0,"risk_level":"none",'
            '"needs_confirmation":false,"decision_summary":"Plan locally."}'
        )
        is True
    )
    assert verifier._valid_router_decision('{"intent":"unknown"}') is False
    assert _routing_error_code(httpx.TimeoutException("timeout")) == "routing_timeout"
    assert _routing_error_code(httpx.ConnectError("connect")) == "routing_connection_error"
    assert _routing_error_code(RuntimeError("other")) == "inference_transport_error"


def test_workflow_verifier_helpers_validate_fake_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    def response(payload: dict[str, object], status: int = 200) -> httpx.Response:
        return httpx.Response(
            status,
            request=httpx.Request("GET", "http://test"),
            json=payload,
        )

    verifier = object.__new__(WorkflowVerifier)
    verifier._client = lambda **_kwargs: _ResponseClient(
        [
            response({"state": "loaded"}),
            response({"state": "unloaded"}),
        ]
    )
    assert verifier._model_load_unload() == "loaded -> unloaded"
    verifier._client = lambda **_kwargs: _ResponseClient([response({"tasks": [{"id": "1"}]})])
    assert verifier._task_listing() == "1 tasks"
    verifier._client = lambda **_kwargs: _ResponseClient(
        [response({"reminder": {"id": "1"}}), response({"reminders": [{}]})]
    )
    assert verifier._reminder_create_list() == "1 reminders"
    verifier._client = lambda **_kwargs: _ResponseClient([response({"voice": {"status": "ok"}})])
    assert verifier._voice_health() == "ok"

    verifier.api_port = 80
    verifier.api_token = "test-token"
    monkeypatch.setattr(
        "apps.runner.verification.services.httpx.get",
        lambda *_args, **_kwargs: response(
            {"ready": True, "tool_worker": {"self_check": True}, "jobs": {"worker_readiness": True}}
        ),
    )
    assert verifier._core_readiness() == "ready with Tool Worker and Job Worker"

    verifier.checks = []
    assert verifier._check("ok", lambda: "done") == "done"
    assert verifier._check("bad", lambda: (_ for _ in ()).throw(RuntimeError("no"))) is None
    assert verifier.checks[-1].ok is False
