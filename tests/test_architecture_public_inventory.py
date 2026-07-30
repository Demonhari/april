from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from types import SimpleNamespace

import click
import pytest
from fastapi.routing import APIWebSocketRoute
from starlette.routing import Mount
from typer.main import get_command

from apps.runner.commands.composition import _CompositionProxy
from apps.runner.main import app as runner_app
from apps.runner.verify import VerifyCheck, _verification_health_failure
from april_common.service_health import ServiceHealthResult
from services.api.server import app as api_app
from services.brain.orchestrator import AprilOrchestrator

EXPECTED_API_ROUTES = {
    ("GET", "/health"),
    ("POST", "/jobs"),
    ("GET", "/jobs"),
    ("GET", "/jobs/{job_id}"),
    ("POST", "/jobs/{job_id}/cancel"),
    ("POST", "/jobs/{job_id}/retry"),
    ("GET", "/diagnostics"),
    ("GET", "/diagnostics/activity"),
    ("GET", "/readiness"),
    ("GET", "/verification/report/latest"),
    ("GET", "/verification/reports"),
    ("GET", "/verification/reports/{report_basename}"),
    ("GET", "/reports"),
    ("GET", "/reports/latest"),
    ("GET", "/reports/latest/{report_type}"),
    ("POST", "/chat"),
    ("POST", "/chat/stream"),
    ("POST", "/voice/input"),
    ("POST", "/wake"),
    ("GET", "/wake/mute"),
    ("POST", "/wake/mute"),
    ("GET", "/sessions"),
    ("POST", "/sessions"),
    ("POST", "/sessions/{session_id}/close"),
    ("POST", "/agents/run"),
    ("POST", "/tools/request"),
    ("POST", "/tools/approve"),
    ("POST", "/tools/deny"),
    ("GET", "/approvals"),
    ("POST", "/memory"),
    ("GET", "/memory/search"),
    ("DELETE", "/memory/{memory_id}"),
    ("GET", "/memory/inspect"),
    ("GET", "/memory/export"),
    ("POST", "/memory/reindex"),
    ("POST", "/memory/repair-index"),
    ("POST", "/feedback"),
    ("GET", "/playbooks"),
    ("POST", "/playbooks/adopt"),
    ("POST", "/playbooks/mine"),
    ("POST", "/playbooks/{playbook_id}/run"),
    ("GET", "/playbooks/{playbook_id}/runs"),
    ("POST", "/playbooks/runs/{run_id}/resume"),
    ("GET", "/evolution/versions"),
    ("POST", "/evolution/rollback"),
    ("GET", "/evolution/adapters"),
    ("POST", "/evolution/adapters/activate"),
    ("POST", "/evolution/adapters/rollback"),
    ("GET", "/evolution/report/latest"),
    ("GET", "/evolution/status"),
    ("GET", "/evolution/history"),
    ("GET", "/evolution/diff"),
    ("POST", "/evolution/off"),
    ("POST", "/evolution/on"),
    ("POST", "/evolution/dataset/export"),
    ("GET", "/evolution/overlays/pending"),
    ("POST", "/evolution/overlays/approve"),
    ("GET", "/evolution/evals/pending"),
    ("GET", "/evolution/evals/pending/{case_id}"),
    ("POST", "/evolution/evals/promote"),
    ("POST", "/evolution/evals/reject"),
    ("GET", "/reminders"),
    ("POST", "/reminders"),
    ("DELETE", "/reminders/{reminder_id}"),
    ("GET", "/tasks"),
    ("GET", "/scheduler/briefing/preview"),
    ("DELETE", "/conversations/{conversation_id}"),
    ("GET", "/projects"),
    ("POST", "/projects"),
    ("POST", "/projects/{project_id}/index"),
    ("POST", "/documents"),
    ("GET", "/documents"),
    ("GET", "/documents/search"),
    ("GET", "/pool/agents"),
    ("GET", "/runtime/models"),
    ("POST", "/runtime/models/load"),
    ("POST", "/runtime/models/unload"),
    ("MOUNT", "/desktop"),
}

EXPECTED_CLI_COMMANDS = {
    line
    for line in """
april
april acceptance
april agent
april agent run
april approvals
april approve
april ask
april audit
april audit verify
april briefing
april chat
april config
april config inspect
april config validate
april conversation
april conversation delete
april database
april database backup
april database check
april database restore
april deny
april desktop
april doctor
april eval
april eval brain
april finetune
april finetune cancel
april finetune doctor
april finetune plan
april finetune status
april go-live
april health
april jobs
april jobs cancel
april jobs list
april jobs retry
april jobs show
april jobs submit
april logs
april memory
april memory delete
april memory doctor
april memory export
april memory list
april memory reindex
april memory repair-index
april memory search
april model
april model benchmark
april model compare-setups
april model doctor
april model download
april model import
april model import-enqueue
april model load
april model profile
april model profile apply
april model profile list
april model recommend
april model unload
april model verify
april models
april package
april package archive
april package build
april package gatekeeper
april package launch-agent
april package launch-agent install
april package launch-agent remove
april package notarize-status
april package notarize-submit
april package sign
april package staple
april package validate
april package validate-release-zip
april package verify-signature
april profile
april profile delete
april profile set
april profile show
april project
april project add
april project index
april projects
april readiness
april reminder
april reminder create
april reminder delete
april reminder list
april reports
april reports clean
april reports latest
april reports list
april reports show
april reports show-latest
april restart
april security
april security credentials
april security credentials migrate
april security credentials rotate
april security memory-encryption
april security memory-encryption provision
april security memory-encryption rotate
april setup
april setup app-stub
april setup bootstrap
april setup checklist
april setup embeddings
april setup mac-activation
april setup models
april setup tokens
april setup voice
april start
april status
april stop
april task
april task list
april verify
april voice
april voice devices
april voice doctor
april voice health
april voice listen
april voice ptt
april voice speaker-gate
april voice speaker-gate disable
april voice speaker-gate enable-soft
april voice test-record
april voice test-stt
april voice test-tts
april voice verify-conversation-live
april voice verify-live
april voice verify-speaker-live
april voice verify-wake-live
doctor
""".splitlines()
    if line
}


def _api_inventory() -> list[tuple[str, str]]:
    inventory: list[tuple[str, str]] = []
    for route in api_app.routes:
        if route.path in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}:
            continue
        if isinstance(route, Mount):
            inventory.append(("MOUNT", route.path))
        elif isinstance(route, APIWebSocketRoute):
            inventory.append(("WEBSOCKET", route.path))
        else:
            inventory.extend((method, route.path) for method in route.methods or ())
    return inventory


def _walk_commands(command: click.Command, prefix: tuple[str, ...] = ()) -> Iterable[str]:
    commands = getattr(command, "commands", None)
    if not isinstance(commands, dict):
        return
    for name, child in sorted(commands.items()):
        path = (*prefix, name)
        yield " ".join(path)
        yield from _walk_commands(child, path)


def _command(root: click.Command, path: tuple[str, ...]) -> click.Command:
    command = root
    for part in path:
        commands = getattr(command, "commands", None)
        assert isinstance(commands, dict)
        command = commands[part]
    return command


def _options(command: click.Command) -> set[str]:
    return {
        option
        for parameter in command.params
        for option in (*getattr(parameter, "opts", ()), *getattr(parameter, "secondary_opts", ()))
    }


def test_api_route_inventory_is_exact_and_has_no_duplicates() -> None:
    inventory = _api_inventory()
    duplicates = {route for route, count in Counter(inventory).items() if count > 1}
    assert duplicates == set()
    assert set(inventory) == EXPECTED_API_ROUTES


def test_cli_command_inventory_is_exact() -> None:
    assert set(_walk_commands(get_command(runner_app))) == EXPECTED_CLI_COMMANDS


def test_cli_compatibility_alias_and_important_options_are_preserved() -> None:
    root = get_command(runner_app)
    expected = {
        ("april", "model", "import"): {
            "--approval-id",
            "--id",
            "--json",
            "--name",
            "--path",
            "--role",
            "--sha256",
            "--verify",
            "--wait",
            "--wait-timeout",
        },
        ("april", "model", "import-enqueue"): {
            "--approval-id",
            "--id",
            "--json",
            "--name",
            "--path",
            "--role",
            "--sha256",
            "--verify",
            "--wait",
            "--wait-timeout",
        },
        ("april", "model", "compare-setups"): {
            "--cooldown-seconds",
            "--json",
            "--shared-model-id",
            "--wait",
            "--wait-timeout",
        },
        ("april", "memory", "reindex"): {
            "--fake",
            "--json",
            "--wait",
            "--wait-timeout",
        },
    }
    assert {path: _options(_command(root, path)) for path in expected} == expected


def test_orchestrator_facade_keeps_public_flow_entry_points() -> None:
    assert AprilOrchestrator.__module__ == "services.brain.orchestrator"
    assert AprilOrchestrator._prepare_turn.__module__.endswith(".context_flow")
    assert AprilOrchestrator._maybe_run_ladder.__module__.endswith(".routing_flow")
    assert AprilOrchestrator.approve_tool.__module__.endswith(".approval_flow")
    assert AprilOrchestrator._finish_message.__module__.endswith(".finalization_flow")


def test_verification_result_shape_and_runtime_failure_reasons_are_compatible() -> None:
    assert asdict(VerifyCheck("runtime health", False, "forbidden")) == {
        "name": "runtime health",
        "ok": False,
        "detail": "forbidden",
        "status": "fail",
    }
    forbidden = ServiceHealthResult(False, 403, "authentication_rejected", "forbidden")
    missing = ServiceHealthResult(False, 404, "endpoint_not_found", "missing")
    assert "authentication was rejected" in _verification_health_failure(
        "http://runtime/runtime/health", "http://api", forbidden
    )
    assert "returned 404" in _verification_health_failure(
        "http://runtime/runtime/health", "http://api", missing
    )


def test_cli_composition_proxy_supports_python_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = SimpleNamespace(
        __spec__=SimpleNamespace(name="apps.runner.main"),
        sentinel="module-entrypoint",
    )
    monkeypatch.delitem(sys.modules, "apps.runner.main")
    monkeypatch.setitem(sys.modules, "__main__", candidate)
    assert _CompositionProxy().sentinel == "module-entrypoint"
