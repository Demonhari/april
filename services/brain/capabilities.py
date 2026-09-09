from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from agents.base import USER_FACING_ASSISTANT_NAME, USER_FACING_IDENTITY_RULE
from agents.registry import AgentRegistry
from april_common.settings import AprilSettings
from services.april_runtime.model_registry import ModelRegistry
from skills.registry import ToolRegistry

_IDENTITY_REQUESTS = {
    "what is your name",
    "what's your name",
    "who are you",
    "tell me your name",
}
_MODEL_ROLES = (
    ("conversation/brain", "brain", "general_agent"),
    ("coding", "coding", "coding_agent"),
    ("reading", "reading", "reading_agent"),
)
_MODEL_STATES = frozenset(
    {"unavailable", "unloaded", "loading", "loaded", "unloading", "error", "unknown"}
)


def _normalized_question(message: str) -> str:
    return " ".join(re.sub(r"[?!.,;:]+", " ", message.casefold()).split())


def is_identity_request(message: str) -> bool:
    """Recognize only narrow, direct requests for APRIL's own name."""

    return _normalized_question(message) in _IDENTITY_REQUESTS


def is_self_introspection_request(message: str) -> bool:
    """Recognize direct APRIL model/runtime status questions.

    This intentionally avoids broad substring matches so a discussion of a
    document, quote, or unrelated model is left to normal routing.
    """

    lowered = _normalized_question(message)
    if lowered in {
        "system status",
        "runtime status",
        "what are you running",
        "what is running",
        "what is your system status",
        "what is your runtime status",
        "what is april's system status",
        "what is aprils system status",
    }:
        return True
    patterns = (
        r"(what|which) models? (are|is) (you|your|april(?:'s|s)) "
        r"(using|running|configured|loaded|available)",
        r"(what|which) models? do you use",
        r"(what|which) models? (are|is) currently "
        r"(using|running|configured|loaded|available)",
        r"(what|which) models? (are|is) "
        r"(configured|loaded|running|available|healthy)",
        r"(what|which) model is (currently )?loaded",
        r"(are|is) (your|april(?:'s|s)) models? "
        r"(loaded|running|available|healthy)",
        r"what is (your|april(?:'s|s)) (current|configured|active|loaded|running) model",
    )
    return any(re.fullmatch(pattern, lowered) is not None for pattern in patterns)


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
    tasks = [
        asyncio.create_task(models_method()),
        asyncio.create_task(health_method(timeout=timeout_seconds)),
    ]
    try:
        models, health = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=timeout_seconds,
        )
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return None
    if not isinstance(models, Mapping) or not isinstance(health, Mapping):
        return None
    raw_models = models.get("models")
    model_items = (
        [item for item in raw_models if isinstance(item, Mapping)]
        if isinstance(raw_models, list)
        else []
    )
    evidence: dict[str, Any] = {
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
    backend = health.get("backend")
    simulated = health.get("simulated")
    if isinstance(backend, str):
        evidence["backend"] = backend
    if isinstance(simulated, bool):
        evidence["simulated"] = simulated
    return evidence


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
                "state=unknown; loaded=unknown; healthy=unknown (runtime evidence unavailable)"
                if runtime_evidence is None
                else "state=unknown; loaded=unknown; healthy=unknown "
                "(runtime did not report this model)"
            )
        else:
            raw_state = evidence.get("state")
            lifecycle_state = (
                raw_state
                if isinstance(raw_state, str) and raw_state in _MODEL_STATES
                else "unknown"
            )
            loaded = (
                "yes"
                if lifecycle_state == "loaded"
                else "no"
                if lifecycle_state in {"unavailable", "unloaded", "error"}
                else "unknown"
            )
            healthy = (
                "yes"
                if lifecycle_state == "loaded"
                and runtime_health == "ok"
                and not evidence.get("load_error")
                else "no"
                if lifecycle_state == "error" or evidence.get("load_error")
                else "unknown"
            )
            state = f"state={lifecycle_state}; loaded={loaded}; healthy={healthy}"
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
    if runtime_evidence is None:
        runtime_note = "Runtime evidence is unavailable; current state and health are unknown."
    elif runtime_evidence.get("simulated") is True:
        runtime_note = (
            "Runtime backend evidence is simulated; it does not prove real GGUF "
            "loading, generation, or quality."
        )
    elif isinstance(runtime_evidence.get("backend"), str):
        runtime_note = (
            f"Runtime backend evidence: {runtime_evidence['backend']}. "
            "This snapshot does not prove generation or quality verification."
        )
    else:
        runtime_note = (
            "Runtime evidence is limited to the local snapshot above; unknown fields "
            "remain unknown."
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


def render_self_status(
    *,
    settings: AprilSettings,
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelRegistry | None = None,
    runtime_evidence: Mapping[str, Any] | None = None,
) -> str:
    """Render concise, application-owned status for a direct self-status query."""

    summary = trusted_capability_summary(
        settings=settings,
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        model_registry=model_registry,
        runtime_evidence=runtime_evidence,
    )
    lines = summary.splitlines()
    start = lines.index("CONFIGURED AI MODELS:") + 1
    end = lines.index("NON-MODEL SUBSYSTEMS:")
    model_lines = lines[start:end]
    if runtime_evidence is None:
        runtime_line = (
            "Current local runtime evidence is unavailable; state and health are unknown."
        )
    elif runtime_evidence.get("simulated") is True:
        runtime_line = (
            "The current runtime evidence is simulated; it does not prove real GGUF "
            "loading or generation."
        )
    else:
        runtime_line = "Loaded and healthy reflect only the current local runtime snapshot."
    return "\n".join(
        [
            "I'm APRIL, your personal local assistant.",
            "Configured AI models:",
            *model_lines,
            "SQLite-backed durable memory is storage, not an AI model.",
            runtime_line,
        ]
    )
