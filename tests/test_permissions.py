from __future__ import annotations

import pytest

from april_common.errors import PermissionDeniedError
from services.permissions.engine import PermissionEngine
from skills.registry import default_registry


def engine() -> PermissionEngine:
    return PermissionEngine(default_registry())


def test_level_1_executes_without_approval(settings_tmp) -> None:
    decision = engine().evaluate(
        tool="read_file", args={"path": str(settings_tmp.home / "README.md")}, agent="coding_agent"
    )
    assert decision.permission_level == 1
    assert decision.confirmation_required is False


def test_level_2_executes_under_policy(settings_tmp) -> None:
    decision = engine().evaluate(tool="create_note", args={"title": "x"}, agent="creative_agent")
    assert decision.permission_level == 2
    assert decision.confirmation_required is False


def test_level_3_blocked_pending_approval(settings_tmp) -> None:
    decision = engine().evaluate(
        tool="write_file", args={"path": str(settings_tmp.home / "x.py")}, agent="coding_agent"
    )
    assert decision.permission_level == 3
    assert decision.confirmation_required is True


def test_level_4_blocked_pending_approval() -> None:
    decision = engine().evaluate(
        tool="open_app", args={"name": "Safari"}, agent="system_action_agent"
    )
    assert decision.permission_level == 4
    assert decision.confirmation_required is True


def test_unknown_tool_denied() -> None:
    with pytest.raises(PermissionDeniedError):
        engine().evaluate(tool="unknown", args={}, agent="coding_agent")


def test_blocked_tool_denied_for_agent(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        engine().evaluate(
            tool="write_file", args={"path": str(settings_tmp.home / "x")}, agent="reading_agent"
        )


def test_model_cannot_lower_permission_level(settings_tmp) -> None:
    decision = engine().evaluate(
        tool="write_file",
        args={"path": str(settings_tmp.home / "x")},
        agent="coding_agent",
        model_permission_level=0,
        model_risk_level="none",
    )
    assert decision.permission_level == 3
