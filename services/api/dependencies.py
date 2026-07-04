from __future__ import annotations

from dataclasses import dataclass

from agents.registry import AgentRegistry
from april_common.audit import AuditLogger
from april_common.config_validation import validate_configuration
from april_common.effective_config import (
    build_agent_registry_from_config,
    build_configured_tool_registry,
    load_permissions_file,
)
from april_common.errors import ConfigError
from april_common.settings import AprilSettings, get_settings
from services.april_runtime.client import RuntimeClient
from services.april_runtime.model_registry import ModelRegistry
from services.brain.orchestrator import AprilOrchestrator
from services.evolution.dreamer import DreamerService
from services.evolution.scheduler import EvolutionSchedulerGate
from services.evolution.versions import PromptOverlayManager
from services.memory.archive import ArchiveReflectionService
from services.memory.database import Database
from services.memory.embeddings import embedding_provider_from_config
from services.memory.migrations import run_migrations
from services.memory.retriever import MemoryRetriever, RuntimeMemoryReranker
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from services.permissions.tool_execution import ToolExecutionService
from services.pool.governor import ResourceGovernor
from services.scheduler import SchedulerService, notification_sink_from_settings
from services.wake.session_manager import SessionManager
from skills.playbooks import PlaybookLoader, PlaybookRunner
from skills.registry import ToolRegistry, default_registry


@dataclass(slots=True)
class ApiContainer:
    settings: AprilSettings
    database: Database
    memory: SqliteMemory
    vector_memory: VectorMemory
    memory_retriever: MemoryRetriever
    runtime_client: RuntimeClient
    tool_registry: ToolRegistry
    permission_engine: PermissionEngine
    approvals: ApprovalStore
    tool_executor: ToolExecutionService
    agent_registry: AgentRegistry
    orchestrator: AprilOrchestrator
    scheduler: SchedulerService | None = None
    session_manager: SessionManager | None = None
    archive_reflection: ArchiveReflectionService | None = None

    def require_session_manager(self) -> SessionManager:
        if self.session_manager is None:
            self.session_manager = SessionManager(
                self.memory,
                continuity_minutes=self.settings.session.continuity_minutes,
                on_close=(
                    self.archive_reflection.reflect_session
                    if self.archive_reflection is not None
                    else None
                ),
            )
        return self.session_manager

    async def aclose(self) -> None:
        """Release every owned resource. Safe to call more than once."""
        if self.scheduler is not None:
            await self.scheduler.stop()
        await self.database.close()


async def build_container(settings: AprilSettings | None = None) -> ApiContainer:
    active_settings = settings or get_settings()
    errors = validate_configuration(active_settings.home)
    if errors:
        raise ConfigError("APRIL configuration is invalid.", {"errors": errors})
    database = Database(active_settings.database_path)
    await database.connect()
    try:
        return await _assemble_container(active_settings, database)
    except BaseException:
        # A failure partway through assembly must not leak the open connection.
        await database.close()
        raise


async def _assemble_container(active_settings: AprilSettings, database: Database) -> ApiContainer:
    await run_migrations(database)
    memory = SqliteMemory(database)
    runtime_client = RuntimeClient(
        active_settings.runtime.url,
        timeout=active_settings.runtime.request_timeout_seconds,
        token=active_settings.runtime.token,
    )
    audit = AuditLogger(active_settings.audit_path)
    embedding = embedding_provider_from_config(
        active_settings.memory.embedding_provider,
        model_id=active_settings.memory.embedding_model_id,
        runtime_client=runtime_client,
        audit=audit,
    )
    vector_memory = VectorMemory(active_settings.vector_index_path, embedding=embedding)
    model_registry = ModelRegistry.from_file(
        active_settings.home / "configs" / "models.yaml",
        root=active_settings.home,
    )
    agent_registry = build_agent_registry_from_config(
        home=active_settings.home,
        model_registry=model_registry,
        tool_registry=default_registry(),
    )
    # Stage-two memory rerank runs through the local runtime with the reading
    # agent's (Scout's) model. When the runtime or model is unavailable the
    # reranker reports failure and retrieval falls back to its deterministic
    # ranking with an audit event — reranking is never faked.
    reading_agent = agent_registry.get("reading_agent")
    reranker = RuntimeMemoryReranker(
        runtime_client,
        model_id=(reading_agent.model_id if reading_agent else None)
        or active_settings.brain.model_id,
    )
    memory_retriever = MemoryRetriever(memory, vector_memory, reranker=reranker, audit=audit)
    tool_registry = build_configured_tool_registry(active_settings.home, agent_registry)
    permissions_config = load_permissions_file(active_settings.home)
    active_settings = active_settings.model_copy(
        update={
            "permissions": active_settings.permissions.model_copy(
                update={"external_actions_enabled": permissions_config.external_actions_enabled}
            )
        }
    )
    permission_engine = PermissionEngine(
        tool_registry,
        approval_required_at=permissions_config.approval_required_at_level,
    )
    approvals = ApprovalStore(
        database,
        audit,
        expiry_seconds=active_settings.permissions.approval_expiry_seconds,
    )
    tool_executor = ToolExecutionService(
        settings=active_settings,
        memory=memory,
        tool_registry=tool_registry,
        permission_engine=permission_engine,
        approvals=approvals,
    )
    overlay_manager = PromptOverlayManager(active_settings, database, audit=audit)
    playbook_loader = PlaybookLoader(active_settings.playbooks_path)
    playbook_runner = PlaybookRunner(tool_executor, memory=memory)
    orchestrator = AprilOrchestrator(
        settings=active_settings,
        runtime_client=runtime_client,
        memory=memory,
        tool_registry=tool_registry,
        permission_engine=permission_engine,
        approvals=approvals,
        tool_executor=tool_executor,
        agent_registry=agent_registry,
        memory_retriever=memory_retriever,
        overlay_manager=overlay_manager,
        playbook_loader=playbook_loader,
        playbook_runner=playbook_runner,
    )
    archive_reflection = ArchiveReflectionService(
        active_settings,
        memory=memory,
        runtime_client=runtime_client,
        vector_memory=vector_memory,
        audit=audit,
    )
    sink = notification_sink_from_settings(active_settings, audit)
    governor = ResourceGovernor(active_settings)
    dreamer = DreamerService(
        active_settings,
        memory=memory,
        gate=EvolutionSchedulerGate(active_settings, memory, governor=governor),
        audit=audit,
    )
    scheduler = SchedulerService(
        settings=active_settings,
        memory=memory,
        audit=audit,
        sink=sink,
        dreamer=dreamer,
    )
    session_manager = SessionManager(
        memory,
        continuity_minutes=active_settings.session.continuity_minutes,
        on_close=archive_reflection.reflect_session,
    )
    return ApiContainer(
        settings=active_settings,
        database=database,
        memory=memory,
        vector_memory=vector_memory,
        memory_retriever=memory_retriever,
        runtime_client=runtime_client,
        tool_registry=tool_registry,
        permission_engine=permission_engine,
        approvals=approvals,
        tool_executor=tool_executor,
        agent_registry=agent_registry,
        orchestrator=orchestrator,
        scheduler=scheduler,
        session_manager=session_manager,
        archive_reflection=archive_reflection,
    )
