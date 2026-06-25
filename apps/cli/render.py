from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def print_jsonish(data: Any) -> None:
    console.print(data)


def print_models(data: dict[str, Any]) -> None:
    table = Table(title="APRIL Models")
    for column in ("id", "role", "state", "missing_path", "backend"):
        table.add_column(column)
    for model in data.get("models", []):
        table.add_row(
            str(model.get("id")),
            str(model.get("role")),
            str(model.get("state")),
            str(model.get("missing_path")),
            str(model.get("backend")),
        )
    console.print(table)


def print_briefing(data: dict[str, Any]) -> None:
    title = str(data.get("title", "APRIL Daily Briefing"))
    console.print(f"[bold]{title}[/bold]")
    console.print(str(data.get("body", "")))


def print_approvals(data: dict[str, Any]) -> None:
    table = Table(title="Pending Approvals")
    for column in ("id", "tool", "permission", "risk", "details", "expires_at"):
        table.add_column(column)
    for item in data.get("approvals", []):
        table.add_row(
            item["id"],
            item["tool"],
            str(item["permission_level"]),
            item["risk_level"],
            _approval_details(item),
            item["expires_at"],
        )
    console.print(table)


def _approval_details(item: dict[str, Any]) -> str:
    """A short, redacted summary of what an approval will do."""
    metadata = item.get("metadata") or {}
    if item.get("tool") == "apply_log_cleanup":
        return (
            f"{metadata.get('candidate_count', '?')} file(s), "
            f"{metadata.get('total_bytes', '?')} bytes under {metadata.get('target', '?')}"
        )
    paths = metadata.get("affected_paths") or item.get("affected_paths") or []
    if paths:
        return f"{len(paths)} path(s)"
    side_effects = item.get("expected_side_effects") or []
    return str(side_effects[0]) if side_effects else ""
