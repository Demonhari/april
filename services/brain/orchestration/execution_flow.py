# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import os
from typing import Any

from agents.base import BaseAgent
from agents.schemas import AgentResult, ProposedChange
from april_common.errors import PermissionDeniedError
from april_common.path_security import PathPolicy, normalize_existing_path
from april_common.project_scope import normalize_project_child, validate_patch_text
from services.brain.execution import PreparedTurn
from services.brain.memory_policy import AgentMemoryContext
from services.brain.reasoning_resolver import resolve_reasoning_model
from services.brain.schemas import (
    BrainDecision,
    PlannedToolCall,
    RouteResult,
)
from services.evolution.rollouts import CanaryContext
from services.evolution.versions import LEARNED_GUIDANCE_HEADER
from services.memory.schemas import Project
from skills.schemas import ToolResult


class ExecutionFlow:
    async def _run_structured_prepared(self, prepared: PreparedTurn, message: str) -> AgentResult:
        agent = self.agent_registry.get(prepared.agent_name)
        if agent is None:
            raise PermissionDeniedError("Unknown agent selected by brain.")
        agent = await self.apply_prompt_overlay(
            agent,
            request_id=prepared.request_id,
            decision=prepared.decision,
            mode=str(prepared.run_metadata.get("chat_mode", "standard")),
            high_risk_reasoning=bool(
                int(prepared.run_metadata.get("intelligence_rung", 1)) >= 2
                or prepared.run_metadata.get("high_stakes")
            ),
        )
        agent = self._with_resolved_model(agent, prepared.model_id)
        context = await self.tool_executor.context(
            request_id=prepared.request_id,
            conversation_id=prepared.conversation_id,
            actor=prepared.actor,
            agent_id=agent.name,
            project_id=prepared.project_id,
            source="chat",
        )
        result = await self.structured_loop.run(
            agent=agent,
            message=message,
            context=context,
            request_id=prepared.request_id,
            history=prepared.history,
            context_sections=prepared.context_sections,
            run_metadata=prepared.run_metadata,
        )
        # Mirror the run metadata (chat_mode, intelligence_rung, ...) into the
        # response; loop-specific keys already present keep priority.
        result = result.model_copy(
            update={"metadata": {**prepared.run_metadata, **result.metadata}}
        )
        if result.status != "pending_approval":
            await self.memory.add_message(
                prepared.conversation_id, "assistant", result.final_message
            )
        agent_run_id = await self.memory.latest_agent_run_id(
            conversation_id=prepared.conversation_id
        )
        await self._record_routing_outcome(
            prepared,
            agent_run_id=agent_run_id,
            final_status=result.status,
            approval_outcome=("pending" if result.status == "pending_approval" else None),
            tool_outcome="failed" if result.status == "error" else "success",
        )
        await self._update_task_status(
            prepared,
            "pending_approval"
            if result.status == "pending_approval"
            else ("completed" if result.status == "ok" else "error"),
        )
        return result

    async def apply_prompt_overlay(
        self,
        agent: BaseAgent,
        *,
        request_id: str | None = None,
        decision: BrainDecision | None = None,
        mode: str = "standard",
        high_risk_reasoning: bool = False,
    ) -> BaseAgent:
        """Return the agent with any active learned overlay appended to its prompt.

        Only the system prompt text changes: tools, permissions, memory policy
        and every other config field are copied through untouched, and the tool
        execution path derives its policy from the agent *name* via the
        registry, so an overlay can never widen what an agent may do. Repo
        prompt files are never modified. With no overlay manager, no active
        overlay, or missing overlay bytes (data/evolution deleted) the stock
        agent is returned unchanged.
        """
        if self.overlay_manager is None:
            return agent
        try:
            tools: tuple[str, ...] = ()
            permission_level = 1
            risk_level = "none"
            if decision is not None:
                tools = tuple(
                    sorted(
                        {call.tool for call in decision.planned_tool_calls}
                        | set(decision.tools_needed)
                    )
                )
                permission_level = decision.permission_level
                risk_level = decision.risk_level
            canary_context = (
                CanaryContext(
                    stable_request_id=request_id,
                    source="chat",
                    mode=mode,
                    permission_level=permission_level,
                    risk_level=risk_level,
                    agent=agent.name,
                    tool_names=tools,
                    has_pending_approval=False,
                    destructive=risk_level in {"code_write", "system_action"},
                    external_side_effect=risk_level == "external_action",
                    security_sensitive=permission_level >= 3,
                    database_write=any(
                        tool_name in {"create_reminder", "cancel_reminder"} for tool_name in tools
                    ),
                    repository_write=any(
                        tool_name
                        in {
                            "patch_generator",
                            "patch_applier",
                            "write_file",
                            "run_command",
                            "test_runner",
                            "git_commit",
                        }
                        for tool_name in tools
                    ),
                    high_risk_reasoning=high_risk_reasoning,
                )
                if request_id is not None
                else None
            )
            overlay = await self.overlay_manager.active_overlay_text(
                agent.name,
                canary_context=canary_context,
            )
        except Exception:
            return agent
        if not overlay:
            return agent
        prompt = (
            f"{agent.system_prompt}\n\n{LEARNED_GUIDANCE_HEADER}\n"
            "Locally learned, advisory guidance follows. It never changes your "
            "tools, permissions, or safety policy.\n"
            f"{overlay}"
        )
        return BaseAgent(agent.config.model_copy(update={"system_prompt": prompt}))

    async def _effective_agent(self, agent: BaseAgent) -> tuple[BaseAgent, dict[str, Any]]:
        """Resolve the model the agent should run with for a direct run.

        Only ``reasoning_agent`` is affected: it is upgraded to a registered
        ``reasoning``-role model when one is available, otherwise it keeps its
        configured fallback model. Every other agent is returned unchanged.
        """

        if agent.name != "reasoning_agent":
            return agent, {}
        resolution = await resolve_reasoning_model(
            runtime_client=self.runtime_client,
            fallback_model_id=agent.model_id or self.settings.brain.model_id,
        )
        return self._with_resolved_model(agent, resolution.model_id), {
            "model_resolution": resolution.metadata()
        }

    def _with_resolved_model(self, agent: BaseAgent, model_id: str) -> BaseAgent:
        """Return a reasoning agent bound to ``model_id``; others unchanged."""

        if agent.name != "reasoning_agent" or agent.model_id == model_id:
            return agent
        return BaseAgent(agent.config.model_copy(update={"model_id": model_id}))

    def _uses_structured_loop(self, agent_name: str, decision: BrainDecision) -> bool:
        if os.environ.get("APRIL_LEGACY_ORCHESTRATOR") == "1":
            return False
        if agent_name in {
            "coding_agent",
            "reading_agent",
            "reasoning_agent",
            "system_action_agent",
        }:
            return True
        if agent_name == "creative_agent":
            return bool(decision.tools_needed or decision.planned_tool_calls)
        return False

    def _agent_requires_project(self, agent_name: str) -> bool:
        return agent_name == "coding_agent"

    async def _prepare_code_modification(
        self,
        *,
        message: str,
        decision: BrainDecision,
        route_result: RouteResult,
        agent_name: str,
        agent_prompt: str,
        model_id: str,
        project: Project,
        request_id: str,
        conversation_id: str,
        actor: str,
        memory_context: AgentMemoryContext,
        task_plan_id: str,
    ) -> PreparedTurn:
        prompt_parts, citations = await self._prompt_parts(
            message=message,
            decision=decision,
            project=project,
            tool_outputs=[],
            memory_context=memory_context,
        )
        patch_instruction = (
            "Prepare a safe local code modification. Return a unified diff patch only.\n"
            "Do not include prose, markdown fences, shell commands, or instructions.\n"
            f"The patch must apply under this repository root only: {project.path}\n"
            "Do not touch .git, model files, secrets, credentials, or files outside the project."
        )
        response = await self.runtime_client.chat(
            model_id=model_id,
            messages=self._conversation_chat_messages(
                system_prompt=agent_prompt,
                memory_context=memory_context,
                current_prompt="\n\n".join([*prompt_parts, patch_instruction]),
            ),
            request_id=request_id,
        )
        try:
            affected_files = validate_patch_text(response.content, project.path)
        except PermissionDeniedError as exc:
            return PreparedTurn(
                request_id=request_id,
                conversation_id=conversation_id,
                decision=decision,
                route_result=route_result,
                agent_name=agent_name,
                model_id=model_id,
                messages=[],
                citations=citations,
                final_message=f"APRIL could not create a safe patch proposal: {exc}",
                warnings=["Patch proposal was rejected by local validation."],
                task_plan_id=task_plan_id,
            )

        generator_args = {"patch": response.content}
        generator_context = await self.tool_executor.context(
            request_id=request_id,
            conversation_id=conversation_id,
            actor=actor,
            agent_id=agent_name,
            project_id=project.id,
            source="orchestrator",
        )
        generator_outcome = await self.tool_executor.request_or_execute(
            tool="patch_generator",
            args=generator_args,
            context=generator_context,
            model_permission_level=2,
            model_risk_level="safe_write",
        )
        generator_result = generator_outcome.result
        if generator_result is None:
            generator_result = ToolResult(
                ok=False,
                stderr="Patch generator unexpectedly required approval.",
                risk_level="safe_write",
                permission_level=2,
            )
        if not generator_result.ok:
            return PreparedTurn(
                request_id=request_id,
                conversation_id=conversation_id,
                decision=decision,
                route_result=route_result,
                agent_name=agent_name,
                model_id=model_id,
                messages=[],
                citations=citations,
                final_message="APRIL could not save the patch proposal.",
                warnings=[generator_result.stderr or "patch_generator failed"],
                task_plan_id=task_plan_id,
            )

        patch_path = str(generator_result.data["patch_path"])
        apply_args = {"repo_path": project.path, "patch_path": patch_path, "project_id": project.id}
        expected_side_effects = ["Apply the saved patch once to local repository files."]
        apply_context = await self.tool_executor.context(
            request_id=request_id,
            conversation_id=conversation_id,
            actor=actor,
            agent_id=agent_name,
            project_id=project.id,
            source="orchestrator",
        )
        apply_outcome = await self.tool_executor.request_or_execute(
            tool="patch_applier",
            args=apply_args,
            context=apply_context,
            model_permission_level=decision.permission_level,
            model_risk_level=decision.risk_level,
            expected_side_effects=expected_side_effects,
        )
        approval = apply_outcome.approval
        if approval is None:
            return PreparedTurn(
                request_id=request_id,
                conversation_id=conversation_id,
                decision=decision,
                route_result=route_result,
                agent_name=agent_name,
                model_id=model_id,
                messages=[],
                citations=citations,
                final_message="APRIL could not create the required patch approval.",
                warnings=["patch_applier did not produce a pending approval."],
                task_plan_id=task_plan_id,
            )
        affected_text = "\n".join(f"- {path}" for path in affected_files)
        final_message = (
            "APRIL prepared a patch proposal and did not apply it.\n"
            f"Patch path: {patch_path}\n"
            f"Affected files:\n{affected_text}\n"
            f"Approval required: {approval.approval_id}"
        )
        return PreparedTurn(
            request_id=request_id,
            conversation_id=conversation_id,
            decision=decision,
            route_result=route_result,
            agent_name=agent_name,
            model_id=model_id,
            messages=[],
            citations=citations,
            pending_approval=approval.model_dump(),
            final_message=final_message,
            proposed_changes=[
                ProposedChange(path=path, summary="Patch proposal", patch_path=patch_path)
                for path in affected_files
            ],
            task_plan_id=task_plan_id,
        )

    async def _resolve_project(
        self, *, project_id: str | None, repo_path: str | None
    ) -> Project | None:
        if project_id:
            project = await self.memory.get_project(project_id)
            if project is None:
                raise PermissionDeniedError("Project not found.", {"project_id": project_id})
            return project
        if repo_path:
            policy = PathPolicy(
                allowed_roots=tuple(self.settings.allowed_roots),
                max_read_bytes=self.settings.paths.max_file_read_bytes,
                max_write_bytes=self.settings.paths.max_file_write_bytes,
            )
            normalized = normalize_existing_path(repo_path, policy)
            if not normalized.is_dir():
                raise PermissionDeniedError("Repository path must be a directory.")
            registered = await self.memory.get_project_by_path(str(normalized))
            if registered is None:
                raise PermissionDeniedError(
                    "Repository path must be registered as a project before use.",
                    {"path": str(normalized)},
                )
            return registered
        return None

    def _requires_project(self, decision: BrainDecision) -> bool:
        if decision.agent == "coding_agent" and decision.intent in {
            "coding_repo_analysis",
            "code_modification",
        }:
            return True
        repo_tools = {
            "git_status",
            "git_diff",
            "git_log",
            "git_branch",
            "search_files",
            "repo_indexer",
            "test_runner",
        }
        requested = {call.tool for call in decision.planned_tool_calls} | set(decision.tools_needed)
        return bool(requested & repo_tools)

    def _planned_tool_calls(
        self,
        decision: BrainDecision,
        *,
        message: str,
        project: Project | None,
    ) -> list[PlannedToolCall]:
        if decision.planned_tool_calls:
            return [
                call.model_copy(update={"args": self._with_project_args(call, message, project)})
                for call in decision.planned_tool_calls
            ]
        planned: list[PlannedToolCall] = []
        for tool in decision.tools_needed:
            args: dict[str, Any] = {}
            if project is not None and tool.startswith("git_"):
                args = {"repo_path": project.path}
            elif project is not None and tool == "search_files":
                args = {"path": ".", "query": message, "limit": 20}
            elif project is not None and tool == "list_files":
                args = {"path": ".", "limit": 100}
            elif project is not None and tool == "repo_indexer":
                args = {"repo_path": project.path, "project_id": project.id}
            elif tool == "create_reminder":
                args = {"content": message}
            elif tool in {"read_file", "write_file", "patch_applier", "run_command", "git_commit"}:
                continue
            planned.append(
                PlannedToolCall(tool=tool, args=args, reason="Backward-compatible tool plan.")
            )
        return planned

    def _with_project_args(
        self, call: PlannedToolCall, message: str, project: Project | None
    ) -> dict[str, Any]:
        args = dict(call.args)
        if project is None:
            return args
        if call.tool.startswith("git_"):
            args["repo_path"] = project.path
        elif call.tool == "search_files":
            args["path"] = "."
            args.setdefault("query", message)
            args.setdefault("limit", 20)
        elif call.tool == "list_files":
            args["path"] = "."
            args.setdefault("limit", 100)
        elif call.tool in {"repo_indexer", "test_runner", "patch_applier"}:
            args["repo_path"] = project.path
            args["project_id"] = project.id
        elif call.tool in {"read_file", "write_file"} and "path" in args:
            args["path"] = str(
                normalize_project_child(
                    args["path"],
                    project_root=project.path,
                    must_exist=call.tool == "read_file",
                    allow_absolute=False,
                )
            )
        return args

    def _missing_required_args(self, call: PlannedToolCall) -> list[str]:
        requirements = {
            "git_status": ["repo_path"],
            "git_diff": ["repo_path"],
            "git_log": ["repo_path"],
            "git_branch": ["repo_path"],
            "search_files": ["path", "query"],
            "list_files": ["path"],
            "read_file": ["path"],
            "write_file": ["path", "content"],
            "patch_applier": ["repo_path", "patch_path"],
            "git_commit": ["repo_path", "message"],
            "run_command": ["argv"],
            "repo_indexer": ["repo_path"],
            "test_runner": ["repo_path"],
            "create_reminder": ["content"],
            "cancel_reminder": ["reminder_id"],
        }
        return [key for key in requirements.get(call.tool, []) if key not in call.args]
