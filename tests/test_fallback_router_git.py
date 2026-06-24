from __future__ import annotations

import pytest

from april_common.effective_config import load_agents_file
from april_common.settings import project_root
from services.brain.fallback_router import FallbackRouter
from skills.registry import default_registry

# Natural-language variations and the read-only git tool each must resolve to.
GIT_CASES = [
    ("git status", "git_status"),
    ("show git status", "git_status"),
    ("show working tree status", "git_status"),
    ("git diff", "git_diff"),
    ("show git diff", "git_diff"),
    ("what changed in this repo", "git_diff"),
    ("git log", "git_log"),
    ("show git log", "git_log"),
    ("recent commits", "git_log"),
    ("git branch", "git_branch"),
    ("current branch", "git_branch"),
    ("list branches", "git_branch"),
]


@pytest.mark.parametrize(("message", "expected_tool"), GIT_CASES)
def test_fallback_routes_git_read_to_correct_tool(message: str, expected_tool: str) -> None:
    decision = FallbackRouter().route(message)
    assert expected_tool in decision.tools_needed
    assert decision.agent == "coding_agent"
    assert decision.permission_level == 1
    assert decision.risk_level == "read_only"
    assert decision.needs_confirmation is False


def test_git_log_and_branch_are_not_mislabelled_as_status() -> None:
    router = FallbackRouter()
    # The original bug routed both of these to git_status.
    assert router.route("show git log").tools_needed == ["git_log"]
    assert router.route("recent commits").tools_needed == ["git_log"]
    assert router.route("list branches").tools_needed == ["git_branch"]
    assert router.route("current branch").tools_needed == ["git_branch"]
    assert "git_status" not in router.route("git log").tools_needed
    assert "git_status" not in router.route("git branch").tools_needed


@pytest.mark.parametrize(("message", "expected_tool"), GIT_CASES)
def test_git_tools_allowed_for_selected_agent_and_level(message: str, expected_tool: str) -> None:
    decision = FallbackRouter().route(message)
    coding_agent = load_agents_file(project_root()).agents["coding_agent"]
    # The tool the router asks for must be permitted for the agent it selected.
    assert expected_tool in coding_agent.allowed_tools
    definition = default_registry().get(expected_tool)
    assert definition is not None
    assert definition.permission_level <= decision.permission_level
    assert definition.risk_level == "read_only"
