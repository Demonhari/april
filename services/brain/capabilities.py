from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from agents.base import USER_FACING_ASSISTANT_NAME, USER_FACING_IDENTITY_RULE
from agents.registry import AgentRegistry
from april_common.settings import AprilSettings
from services.april_runtime.model_registry import ModelRegistry
from skills.registry import ToolRegistry

_SELF_INTROSPECTION_TERMS = (
    "what models",
    "which models",
    "models are you",
    "system status",
    "runtime status",
    "what are you running",
    "what is running",
)
_MODEL_ROLES = (
    ("conversation/brain", "brain", "general_agent"),
    ("coding", "coding", "coding_agent"),
    ("reading", "reading", "reading_agent"),
)


def is_self_introspection_request(message: str) -> bool:
    lowered = " ".join(message.casefold().split())
    return any(term in lowered for term in _SELF_INTROSPECTION_TERMS)


async def collect_runtime_self_evidence(
    runtime_client: object,
    *,
    timeout_seconds: float = 1.5,
) -> dict[str, Any] | None:
    """Read bounded local runtime state for explicit self-status questions.

    Runtime state is advisory context only. Missing methods, malformed payloads,
    timeouts, and an offline Runtime all degrade to unknown state rather than
    changing the response path or making a readiness claim.
    """

    models_method = getattr(runtime_client, "models", None)
    health_method = getattr(runtime_client, "health", None)
    if not callable(models_method) or not callable(health_method):
        return None
    try:
        models, health = await asyncio.wait_for(
            asyncio.gather(
                models_method(),
                health_method(timeout=timeout_seconds),
            ),
            timeout=timeout_seconds,
        )
    except Exception:
        return None
    if not isinstance(models, Mapping) or not isinstance(health, Mapping):
        return None
    raw_models = models.get("models")
    model_items = (
        [item for item in raw_models if isinstance(item, Mapping)]
        if isinstance(raw_models, list)
        else []
    )
    return {
        "models": [
            {
                "id": item.get("id"),
                "state": item.get("state"),
                "load_error": item.get("load_error"),
            }
            for item in model_items
            if isinstance(item.get("id"), str)
        ],
        "health_status": (health.get("status") if isinstance(health.get("status"), str) else None),
    }


def trusted_capability_summary(
    *,
    settings: AprilSettings,
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelRegistry | None = None,
    runtime_evidence: Mapping[str, Any] | None = None,
) -> str:
    """Return allow-listed, application-owned APRIL self/context facts.

    Registry model definitions provide configured role IDs. Runtime evidence is
    optional and is only included when an explicit self-introspection request
    asked the local Runtime for bounded status. Paths, credentials, arbitrary
    YAML, and source-inspection claims are deliberately omitted.
    """

    by_name = {agent.name: agent for agent in agent_registry.list()}
    configured_registry = model_registry
    if configured_registry is None:
        try:
            configured_registry = ModelRegistry.from_file(
                settings.home / "configs" / "models.yaml", root=settings.home
            )
        except Exception:
            configured_registry = None

    known_roles = {role for _, role, _ in _MODEL_ROLES}
    models_by_role = (
        {
            str(model.role): model
            for model in configured_registry.list()
            if model.role in known_roles
        }
        if configured_registry is not None
        else {}
    )
    runtime_models = {
        str(item["id"]): item
        for item in (runtime_evidence or {}).get("models", [])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    runtime_health = (runtime_evidence or {}).get("health_status")

    def configured_model_line(label: str, role: str, agent_name: str) -> str:
        definition = models_by_role.get(role)
        agent = by_name.get(agent_name)
        model_id = definition.id if definition is not None else (agent.model_id if agent else None)
        if model_id is None:
            return f"- {label}: unavailable in trusted configuration"
        evidence = runtime_models.get(model_id)
        if evidence is None:
            state = (
                "loaded=unknown; healthy=unknown (runtime status not queried)"
                if runtime_evidence is None
                else "loaded=unknown; healthy=unknown (runtime did not report this model)"
            )
        else:
            raw_state = evidence.get("state")
            loaded = (
                "yes"
                if raw_state == "loaded"
                else "no"
                if raw_state in {"unavailable", "unloaded", "error"}
                else "unknown"
            )
            healthy = (
                "yes"
                if raw_state == "loaded"
                and runtime_health == "ok"
                and not evidence.get("load_error")
                else "no"
                if raw_state == "error" or evidence.get("load_error")
                else "unknown"
            )
            state = f"loaded={loaded}; healthy={healthy}"
        return f"- {label}: {model_id} (configured; {state})"

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
    runtime_note = (
        "Runtime evidence was unavailable; loaded and healthy are unknown."
        if runtime_evidence is None
        else (
            "Runtime evidence is limited to the local Runtime snapshot above; "
            "unknown fields must remain unknown."
        )
    )
    return "\n".join(
        [
            "[TRUSTED APRIL SELF CONTEXT]",
            "IDENTITY:",
            f"- User-facing assistant name: {USER_FACING_ASSISTANT_NAME}.",
            f"- {USER_FACING_IDENTITY_RULE}",
            "- Internal pool call signs are implementation metadata, not alternate "
            "user identities.",
            "CONFIGURED AI MODELS:",
            *[
                configured_model_line(label, role, agent_name)
                for label, role, agent_name in _MODEL_ROLES
            ],
            "NON-MODEL SUBSYSTEMS:",
            "- SQLite-backed durable memory is local storage, not an AI model.",
            f"- Local runtime backend: {settings.runtime.backend}.",
            "- Approval-controlled local tools are configured for scoped "
            "repository/file inspection, "
            "document work, memory/reminders, and code/test actions: "
            f"{', '.join(local_tools)}.",
            "- Archive is an internal closed-session reflection component, not an "
            "interactive chat agent.",
            "STATUS DISCIPLINE:",
            f"- {runtime_note}",
            "- Tool use remains subject to project roots, deterministic permissions, "
            "exact approvals, and audit checks.",
            "- Do not claim source inspection, tool execution, model loading, or "
            "model health unless corresponding evidence is supplied.",
        ]
    )
