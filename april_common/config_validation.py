from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from april_common.effective_config import (
    build_agent_registry_from_config,
    build_configured_tool_registry,
    load_agents_file,
    load_permissions_file,
    load_tools_file,
)
from april_common.errors import ConfigError
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelRegistry
from skills.registry import default_registry


def validate_configuration(home: Path) -> list[str]:
    root = home.expanduser().resolve()
    errors: list[str] = []
    try:
        settings = load_settings(root=root)
    except Exception as exc:
        errors.append(f"configs/april.yaml: {exc}")
        settings = None
    try:
        model_registry = ModelRegistry.from_file(root / "configs" / "models.yaml", root=root)
    except ConfigError as exc:
        errors.append(f"configs/models.yaml: {exc}")
        model_registry = None
    base_tools = default_registry()
    agent_registry = None
    try:
        load_tools_file(root)
    except (ConfigError, ValidationError) as exc:
        errors.append(f"configs/tools.yaml: {exc}")
    try:
        load_permissions_file(root)
    except (ConfigError, ValidationError) as exc:
        errors.append(f"configs/permissions.yaml: {exc}")
    try:
        load_agents_file(root)
        if model_registry is not None:
            agent_registry = build_agent_registry_from_config(
                home=root,
                model_registry=model_registry,
                tool_registry=base_tools,
            )
    except (ConfigError, ValidationError) as exc:
        errors.append(f"configs/agents.yaml: {exc}")
    if agent_registry is not None:
        try:
            build_configured_tool_registry(root, agent_registry)
        except (ConfigError, ValidationError) as exc:
            errors.append(f"configs/tools.yaml: {exc}")
    if settings is not None and settings.api.host != "127.0.0.1":
        errors.append("configs/april.yaml: API host should default to 127.0.0.1")
    if settings is not None and settings.runtime.host != "127.0.0.1":
        errors.append("configs/april.yaml: Runtime host should default to 127.0.0.1")
    return errors
