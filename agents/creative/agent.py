from __future__ import annotations

from pathlib import Path

from agents.base import BaseAgent, load_prompt
from agents.schemas import AgentConfig


def creative_agent() -> BaseAgent:
    prompt_path = Path(__file__).with_name("prompt.md")
    return BaseAgent(
        AgentConfig(
            name="creative_agent",
            description="Emails, ideas, scripts, and concepts.",
            model_id="april-brain",
            system_prompt_path=str(prompt_path),
            allowed_tools={"create_note", "search_notes"},
            blocked_tools={"send_email", "open_url"},
            memory_access_policy="conversation_and_safe_memory",
            maximum_tool_iterations=3,
            system_prompt=load_prompt(prompt_path),
        )
    )
