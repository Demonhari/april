from __future__ import annotations

from dataclasses import dataclass

from agents.registry import AgentRegistry, default_agent_registry
from april_common.audit import AuditLogger
from april_common.config_validation import validate_configuration
from april_common.errors import ConfigError
from april_common.settings import AprilSettings, get_settings
from services.april_runtime.client import RuntimeClient
from services.brain.orchestrator import AprilOrchestrator
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.retriever import MemoryRetriever
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from services.permissions.tool_execution import ToolExecutionService
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


async def build_container(settings: AprilSettings | None = None) -> ApiContainer:
    active_settings = settings or get_settings()
    errors = validate_configuration(active_settings.home)
    if errors:
        raise ConfigError("APRIL configuration is invalid.", {"errors": errors})
    database = Database(active_settings.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    vector_memory = VectorMemory(active_settings.vector_index_path)
    memory_retriever = MemoryRetriever(memory, vector_memory)
    runtime_client = RuntimeClient(
        active_settings.runtime.url,
        timeout=active_settings.runtime.request_timeout_seconds,
    )
    tool_registry = default_registry()
    permission_engine = PermissionEngine(tool_registry)
    audit = AuditLogger(active_settings.audit_path)
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
    agent_registry = default_agent_registry()
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
    )
