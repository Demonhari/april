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

    violations = {
        node.name: _span(node)
        for node in ast.walk(_module("apps/runner/main.py"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _span(node) >= 100
    }
    assert violations == {}


def test_orchestrator_facade_does_not_gain_large_workflows() -> None:
    """Implementation growth belongs in the existing brain service modules."""

    violations = {
        node.name: _span(node)
        for node in ast.walk(_module("services/brain/orchestrator.py"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _span(node) >= 100
    }
    assert violations == {}


def test_verification_composition_root_does_not_grow() -> None:
    """Focused verification modules should absorb future checks."""

    limits = {
        "services/api/server.py": 100,
        "services/brain/orchestrator.py": 250,
        "apps/runner/main.py": 250,
        "apps/runner/verify.py": 700,
    }
    actual = {
        path: len((REPOSITORY_ROOT / path).read_text(encoding="utf-8").splitlines())
        for path in limits
    }
    assert {path: count for path, count in actual.items() if count > limits[path]} == {}


def _import_targets(path: Path) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            targets.add(node.module)
    return targets


def test_extracted_modules_respect_import_boundaries() -> None:
    route_modules = (REPOSITORY_ROOT / "services/api/routes").glob("*.py")
    assert all("services.api.server" not in _import_targets(path) for path in route_modules)

    orchestration_modules = (REPOSITORY_ROOT / "services/brain/orchestration").glob("*.py")
    assert all(
        not any(target.startswith("apps.runner") for target in _import_targets(path))
        for path in orchestration_modules
    )

    service_modules = (REPOSITORY_ROOT / "services").rglob("*.py")
    assert all(
        not any(target.startswith("apps.desktop") for target in _import_targets(path))
        for path in service_modules
    )


def test_verification_modules_have_no_import_time_work() -> None:
    """Verification modules may define objects, but must not execute checks on import."""

    violations: list[str] = []
    for path in (REPOSITORY_ROOT / "apps/runner/verification").glob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                violations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if isinstance(value, ast.Call):
                    violations.append(f"{path.name}:{node.lineno}")
    assert violations == []


def test_extracted_implementation_modules_have_reasonable_size() -> None:
    """Prevent simply moving each old monolith into one replacement file."""

    roots = (
        REPOSITORY_ROOT / "services/api",
        REPOSITORY_ROOT / "services/brain/orchestration",
        REPOSITORY_ROOT / "apps/runner/commands",
        REPOSITORY_ROOT / "apps/runner/verification",
    )
    oversized = {
        str(path.relative_to(REPOSITORY_ROOT)): len(path.read_text(encoding="utf-8").splitlines())
        for root in roots
        for path in root.glob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > 900
    }
    assert oversized == {}
