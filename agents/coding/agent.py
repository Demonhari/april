from __future__ import annotations

from pathlib import Path

from agents.base import BaseAgent, load_prompt
from agents.schemas import AgentConfig


def coding_agent() -> BaseAgent:
    prompt_path = Path(__file__).with_name("prompt.md")
    return BaseAgent(
        AgentConfig(
            name="coding_agent",
            description="Repository inspection, code explanation, and patch proposals.",
            model_id="april-coding",
            system_prompt_path=str(prompt_path),
            allowed_tools={
                "git_status",
                "git_diff",
                "git_log",
                "git_branch",
                "list_files",
                "read_file",
                "search_files",
                "repo_indexer",
                "patch_generator",
                "patch_applier",
                "test_runner",
                "write_file",
                "git_commit",
                "run_command",
            },
            blocked_tools={"git_push", "open_url", "open_app"},
            memory_access_policy="project_memory",
            maximum_tool_iterations=5,
            system_prompt=load_prompt(prompt_path),
        )
    )
