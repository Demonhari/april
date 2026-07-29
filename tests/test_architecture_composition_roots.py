from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "websocket"}


def _module(path: str) -> ast.Module:
    return ast.parse((REPOSITORY_ROOT / path).read_text(encoding="utf-8"))


def _span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    assert node.end_lineno is not None
    return node.end_lineno - node.lineno + 1


def test_api_composition_root_contains_no_endpoint_handlers() -> None:
    """New HTTP handlers belong in ``services.api.routes``, not server.py."""

    violations: list[str] = []
    for node in ast.walk(_module("services/api/server.py")):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr in HTTP_METHODS
            ):
                violations.append(f"{node.name}:{node.lineno}")
    assert violations == []


def test_cli_composition_root_does_not_gain_large_command_implementations() -> None:
    """Large CLI workflows must move to ``apps.runner.commands``."""

    grandfathered = {
        "_doctor",
        "_memory_doctor_report",
        "setup_mac_activation",
        "verify",
        "acceptance",
    }
    violations = {
        node.name: _span(node)
        for node in ast.walk(_module("apps/runner/main.py"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _span(node) >= 100
        and node.name not in grandfathered
    }
    assert violations == {}


def test_orchestrator_facade_does_not_gain_large_workflows() -> None:
    """Implementation growth belongs in the existing brain service modules."""

    grandfathered = {
        "stream_chat",
        "_prepare_turn",
        "approve_tool",
        "_prepare_code_modification",
    }
    violations = {
        node.name: _span(node)
        for node in ast.walk(_module("services/brain/orchestrator.py"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _span(node) >= 100
        and node.name not in grandfathered
    }
    assert violations == {}


def test_verification_composition_root_does_not_grow() -> None:
    """Focused verification modules should absorb future checks."""

    verify_path = REPOSITORY_ROOT / "apps/runner/verify.py"
    assert len(verify_path.read_text(encoding="utf-8").splitlines()) <= 3_400
