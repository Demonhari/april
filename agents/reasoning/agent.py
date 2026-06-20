from __future__ import annotations

from pathlib import Path

from agents.base import BaseAgent, load_prompt
from agents.schemas import AgentConfig


def reasoning_agent() -> BaseAgent:
    prompt_path = Path(__file__).with_name("prompt.md")
    return BaseAgent(
        AgentConfig(
            name="reasoning_agent",
            description="Architecture decisions and deeper analysis when configured.",
            model_id=None,
            system_prompt_path=str(prompt_path),
            allowed_tools={"read_file", "search_files", "git_status", "git_diff"},
            blocked_tools={"write_file", "run_command", "git_commit"},
            memory_access_policy="conversation_and_safe_memory",
            maximum_tool_iterations=5,
            system_prompt=load_prompt(prompt_path),
        )
    )
