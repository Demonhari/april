from __future__ import annotations

from pathlib import Path

from agents.base import BaseAgent, load_prompt
from agents.schemas import AgentConfig


def reading_agent() -> BaseAgent:
    prompt_path = Path(__file__).with_name("prompt.md")
    return BaseAgent(
        AgentConfig(
            name="reading_agent",
            description="Document and chunk summarization.",
            model_id="april-reading",
            system_prompt_path=str(prompt_path),
            allowed_tools={"read_file", "search_files", "document_search", "document_indexer"},
            blocked_tools={
                "write_file",
                "run_command",
                "git_commit",
                "patch_applier",
                "open_url",
                "open_app",
            },
            memory_access_policy="conversation_and_safe_memory",
            maximum_tool_iterations=3,
            system_prompt=load_prompt(prompt_path),
        )
    )
