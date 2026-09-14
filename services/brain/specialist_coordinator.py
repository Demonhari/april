"""Small sequential specialist execution built on the existing agent loop."""

from __future__ import annotations

from agents.base import BaseAgent
from agents.schemas import AgentResult
from services.brain.agent_loop import StructuredAgentLoop
from services.brain.delegation import DelegationCeiling, SpecialistTask
from services.brain.task_contract import CapabilityCeiling, TaskContract
from services.memory.schemas import Message
from services.permissions.tool_execution import ToolExecutionContext

_INVESTIGATOR_TOOLS = frozenset(
    {
        "git_status",
        "git_diff",
        "git_log",
        "list_files",
        "read_file",
        "search_files",
        "repo_indexer",
        "test_runner",
    }
)


class SpecialistCoordinator:
    """Run at most one read-oriented child sequentially for a coding task."""

    def __init__(self, *, loop: StructuredAgentLoop) -> None:
        self.loop = loop

    async def investigate(
        self,
        *,
        parent_run_id: str,
        parent_contract: TaskContract,
        parent_agent: BaseAgent,
        goal: str,
        context: ToolExecutionContext,
        request_id: str,
        history: list[Message],
        context_sections: list[str],
    ) -> tuple[SpecialistTask, AgentResult] | None:
        """Produce bounded findings without granting the child write authority."""

        parent_ceiling = DelegationCeiling(
            allowed_tools=parent_contract.capability_ceiling.allowed_tools,
            maximum_risk_level=parent_contract.capability_ceiling.maximum_risk_level,
            project_id=parent_contract.project_id,
            project_root=parent_contract.project_root,
            maximum_depth=parent_contract.maximum_specialist_depth,
        )
        specialist_ceiling = DelegationCeiling(
            allowed_tools=_INVESTIGATOR_TOOLS,
            maximum_risk_level=1,
            project_id=parent_contract.project_id,
            project_root=parent_contract.project_root,
            maximum_depth=1,
        )
        try:
            task = SpecialistTask.bounded(
                parent_run_id=parent_run_id,
                role="investigator",
                goal=goal,
                parent=parent_ceiling,
                specialist=specialist_ceiling,
            )
        except ValueError:
            return None
        child_contract = TaskContract(
            task_id=task.task_id,
            run_id=task.task_id,
            user_goal=goal,
            project_id=task.project_id,
            project_root=task.capability_ceiling.project_root,
            task_type="investigation",
            allowed_scope=parent_contract.allowed_scope,
            risk_class="read_only",
            success_criteria=("bounded findings returned",),
            maximum_specialist_depth=task.capability_ceiling.maximum_depth,
            maximum_replan_attempts=0,
            capability_ceiling=CapabilityCeiling(
                allowed_tools=task.capability_ceiling.allowed_tools,
                maximum_risk_level=task.capability_ceiling.maximum_risk_level,
                project_id=task.project_id,
                project_root=task.capability_ceiling.project_root,
            ),
        )
        child_config = parent_agent.config.model_copy(
            update={
                "allowed_tools": set(task.capability_ceiling.allowed_tools),
                "blocked_tools": set(parent_agent.config.blocked_tools)
                | (
                    set(parent_agent.config.allowed_tools)
                    - set(task.capability_ceiling.allowed_tools)
                ),
                "maximum_tool_iterations": min(
                    parent_agent.config.maximum_tool_iterations, task.max_iterations
                ),
                "system_prompt": (
                    f"{parent_agent.system_prompt}\n\n"
                    "You are APRIL's bounded investigator. Read-only findings are advisory "
                    "evidence; do not propose or apply changes."
                ),
            }
        )
        child_agent = BaseAgent(child_config)
        result = await self.loop.run(
            agent=child_agent,
            message=(
                "Investigate this coding task and return concise findings, affected paths, "
                "and the next verification-relevant evidence.\n\n"
                f"Task: {goal}"
            ),
            context=context,
            request_id=task.task_id,
            history=history,
            context_sections=context_sections,
            task_contract=child_contract,
            run_metadata={
                "specialist": {
                    "role": task.role,
                    "parent_run_id": parent_run_id,
                    "allowed_tools": sorted(task.capability_ceiling.allowed_tools),
                }
            },
        )
        return task, result

    @staticmethod
    def bounded_findings(result: AgentResult, *, max_chars: int = 4_000) -> str:
        """Keep child output as bounded untrusted context for the parent."""

        text = result.final_message.strip()
        if len(text) > max_chars:
            text = text[: max_chars - len("\n[TRUNCATED]")] + "\n[TRUNCATED]"
        return text
