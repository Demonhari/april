from __future__ import annotations

from agents.registry import AgentRegistry
from april_common.settings import AprilSettings
from skills.registry import ToolRegistry


def trusted_capability_summary(
    *,
    settings: AprilSettings,
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
) -> str:
    """Return a small allow-listed capability description for answer prompts.

    This is application-owned context, not a claim that every configured model
    is currently loaded or healthy. It deliberately omits paths, credentials,
    arbitrary YAML, and source-inspection claims.
    """

    by_name = {agent.name: agent for agent in agent_registry.list()}
    general = by_name.get("general_agent")
    coding = by_name.get("coding_agent")
    reading = by_name.get("reading_agent")
    model_label = general.model_id if general is not None else None
    coding_label = coding.model_id if coding is not None else None
    reading_label = reading.model_id if reading is not None else None
    configured_tools = {tool.name for tool in tool_registry.list()}
    local_tools = sorted(
        configured_tools
        & {
            "read_file",
            "search_files",
            "list_files",
            "git_status",
            "git_diff",
            "git_log",
            "repo_indexer",
            "document_search",
            "document_indexer",
            "remember_memory",
            "create_reminder",
            "list_reminders",
            "cancel_reminder",
            "patch_generator",
            "patch_applier",
            "test_runner",
            "run_command",
        }
    )
    return (
        "[TRUSTED APRIL CAPABILITY SUMMARY]\n"
        f"APRIL is a local-first assistant using the configured local runtime "
        f"backend ({settings.runtime.backend}). Its resident conversation brain "
        f"is configured as {model_label or 'unavailable'}; coding assistance is "
        f"configured as {coding_label or 'unavailable'}; document/reading assistance "
        f"is configured as {reading_label or 'unavailable'}. Local durable memory "
        "is available through policy-controlled SQLite-backed storage. "
        "Approval-controlled local tools are configured for scoped repository/file "
        "inspection, document work, memory/reminders, and code/test actions: "
        f"{', '.join(local_tools)}. "
        "Tool use remains subject to project roots, deterministic permissions, exact "
        "approvals, and audit checks. Configured does not mean loaded, healthy, or "
        "available for this turn; source inspection and tool execution must not be "
        "claimed unless their results are supplied. Archive is an internal session "
        "reflection component, not an interactive chat agent."
    )
