from __future__ import annotations

from pathlib import Path

from agents.base import BaseAgent, load_prompt
from agents.schemas import AgentConfig


def general_agent() -> BaseAgent:
    prompt_path = Path(__file__).with_name("prompt.md")
    return BaseAgent(
        AgentConfig(
            name="general_agent",
            description="Chat, planning, and small answers.",
            model_id="april-brain",
            system_prompt_path=str(prompt_path),
            allowed_tools=set(),
            blocked_tools=set(),
            memory_access_policy="conversation_and_safe_memory",
            maximum_tool_iterations=5,
            system_prompt=load_prompt(prompt_path),
        )
    )
