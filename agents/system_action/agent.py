from __future__ import annotations

from pathlib import Path

from agents.base import BaseAgent, load_prompt
from agents.schemas import AgentConfig


def system_action_agent() -> BaseAgent:
    prompt_path = Path(__file__).with_name("prompt.md")
    return BaseAgent(
        AgentConfig(
            name="system_action_agent",
            description="Tightly constrained local system actions.",
            model_id="april-brain",
            system_prompt_path=str(prompt_path),
            allowed_tools={"list_files", "run_command", "open_app", "open_url"},
            blocked_tools={"git_push", "deploy", "send_email", "payment"},
            memory_access_policy="none",
            maximum_tool_iterations=2,
            system_prompt=load_prompt(prompt_path),
        )
    )
