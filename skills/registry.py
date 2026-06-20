from __future__ import annotations

from typing import Any

from skills.schemas import ToolDefinition, ToolResult


async def unavailable_executor(args: dict[str, Any]) -> ToolResult:
    return ToolResult(
        ok=False,
        stderr="Capability is registered as unavailable in the MVP.",
        risk_level="external_action",
        permission_level=5,
    )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Duplicate tool: {definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    async def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        definition = self.get(name)
        if definition is None:
            return ToolResult(
                ok=False,
                stderr=f"Unknown tool: {name}",
                risk_level="external_action",
                permission_level=5,
            )
        return await definition.executor(args)


def default_registry() -> ToolRegistry:
    from skills.apps.open_app import open_app_definition
    from skills.apps.open_url import open_url_definition
    from skills.code.patch_applier import patch_applier_definition
    from skills.code.patch_generator import patch_generator_definition
    from skills.code.repo_indexer import repo_indexer_definition
    from skills.code.test_runner import test_runner_definition
    from skills.filesystem.list_files import list_files_definition
    from skills.filesystem.read_file import read_file_definition
    from skills.filesystem.search_files import search_files_definition
    from skills.filesystem.write_file import write_file_definition
    from skills.git.git_branch import git_branch_definition
    from skills.git.git_commit import git_commit_definition
    from skills.git.git_diff import git_diff_definition
    from skills.git.git_log import git_log_definition
    from skills.git.git_status import git_status_definition
    from skills.notes.create_note import create_note_definition
    from skills.notes.search_notes import search_notes_definition
    from skills.reminders.create_reminder import create_reminder_definition
    from skills.reminders.list_reminders import list_reminders_definition
    from skills.terminal.run_command import run_command_definition

    registry = ToolRegistry()
    for definition in (
        list_files_definition(),
        read_file_definition(),
        search_files_definition(),
        write_file_definition(),
        git_status_definition(),
        git_diff_definition(),
        git_log_definition(),
        git_branch_definition(),
        git_commit_definition(),
        run_command_definition(),
        repo_indexer_definition(),
        patch_generator_definition(),
        patch_applier_definition(),
        test_runner_definition(),
        create_note_definition(),
        search_notes_definition(),
        create_reminder_definition(),
        list_reminders_definition(),
        open_app_definition(),
        open_url_definition(),
    ):
        registry.register(definition)
    registry.register(
        ToolDefinition(
            name="git_push",
            description="Unavailable external Git push capability.",
            permission_level=5,
            risk_level="external_action",
            confirmation_required=True,
            allowed_agents={"system_action_agent"},
            executor=unavailable_executor,
        )
    )
    return registry
