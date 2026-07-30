from __future__ import annotations

import uuid

from agents.registry import AgentRegistry
from agents.schemas import AgentResult
from april_common.settings import AprilSettings
from services.april_runtime.client import RuntimeClient
from services.brain.agent_loop import StructuredAgentLoop
from services.brain.conversation_context import ConversationContextService
from services.brain.intelligence_ladder import (
    ChatMode,
    IntelligenceLadder,
)
from services.brain.orchestration.approval_flow import ApprovalFlow
from services.brain.orchestration.context_flow import ContextFlow
from services.brain.orchestration.execution_flow import ExecutionFlow
from services.brain.orchestration.finalization_flow import FinalizationFlow
from services.brain.orchestration.interaction_flow import InteractionFlow
from services.brain.orchestration.routing_flow import RoutingFlow
from services.brain.router import BrainRouter
from services.brain.routing_reliability import RoutingReliabilityService
from services.evolution.versions import PromptOverlayManager
from services.memory.retriever import MemoryRetriever
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from services.permissions.tool_execution import ToolExecutionService
from services.pool.agent_pool import AgentPool
from skills.playbooks.loader import PlaybookLoader
from skills.playbooks.runner import PlaybookRunner
from skills.registry import ToolRegistry


class AprilOrchestrator(
    InteractionFlow,
    RoutingFlow,
    ContextFlow,
    ApprovalFlow,
    ExecutionFlow,
    FinalizationFlow,
):
    def __init__(
        self,
        *,
        settings: AprilSettings,
        runtime_client: RuntimeClient,
        memory: SqliteMemory,
        tool_registry: ToolRegistry,
        permission_engine: PermissionEngine,
        approvals: ApprovalStore,
        tool_executor: ToolExecutionService,
        agent_registry: AgentRegistry,
        memory_retriever: MemoryRetriever | None = None,
        brain_router: BrainRouter | None = None,
        overlay_manager: PromptOverlayManager | None = None,
        playbook_loader: PlaybookLoader | None = None,
        playbook_runner: PlaybookRunner | None = None,
        agent_pool: AgentPool | None = None,
    ) -> None:
        self.settings = settings
        self.runtime_client = runtime_client
        self.memory = memory
        self.tool_registry = tool_registry
        self.permission_engine = permission_engine
        self.approvals = approvals
        self.tool_executor = tool_executor
        self.agent_registry = agent_registry
        self.memory_retriever = memory_retriever
        self.overlay_manager = overlay_manager
        self.playbook_loader = playbook_loader
        self.playbook_runner = playbook_runner
        self.agent_pool = agent_pool
        self.brain_router = brain_router or BrainRouter(
            runtime_client,
            brain_model_id=settings.brain.model_id,
            router_model_id=settings.brain.router_model_id,
        )
        self.routing_reliability = RoutingReliabilityService(
            memory.database,
            settings.brain,
        )
        self.structured_loop = StructuredAgentLoop(
            runtime_client=runtime_client,
            tool_executor=tool_executor,
            memory=memory,
            context_settings=settings.conversation_context,
        )
        self.conversation_context = ConversationContextService(
            memory=memory,
            runtime_client=runtime_client,
            agent_registry=agent_registry,
            settings=settings.conversation_context,
            audit=approvals.audit,
        )
        self.intelligence_ladder = IntelligenceLadder(
            settings=settings,
            runtime_client=runtime_client,
            agent_registry=agent_registry,
        )

    async def chat(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        request_id: str | None = None,
        actor: str = "local-user",
        project_id: str | None = None,
        repo_path: str | None = None,
        mode: ChatMode = "standard",
    ) -> AgentResult:
        await self._maybe_record_implicit_correction(message, conversation_id)
        reminder_reflex = await self._maybe_direct_reminder_reflex(
            message,
            conversation_id=conversation_id,
            request_id=request_id or str(uuid.uuid4()),
            actor=actor,
            project_id=project_id,
            repo_path=repo_path,
        )
        if reminder_reflex is not None:
            return reminder_reflex
        playbook_result = await self._maybe_run_playbook(
            message,
            conversation_id=conversation_id,
            request_id=request_id or str(uuid.uuid4()),
            actor=actor,
            project_id=project_id,
        )
        if playbook_result is not None:
            return playbook_result
        prepared = await self._prepare_turn(
            message,
            conversation_id=conversation_id,
            request_id=request_id,
            actor=actor,
            project_id=project_id,
            repo_path=repo_path,
            structured_specialists=True,
            mode=mode,
        )
        selection = self._select_intelligence_rung(prepared, message=message, mode=mode)
        self._schedule_agent_prewarm(prepared)
        ladder_result = await self._maybe_run_ladder(prepared, message, selection)
        if ladder_result is not None:
            return ladder_result
        if selection.rung == 2:
            return await self._run_verified_prepared(prepared, message)
        return await self._run_standard_prepared(prepared, message)
