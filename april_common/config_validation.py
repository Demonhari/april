from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agents.registry import default_agent_registry
from april_common.errors import ConfigError
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelRegistry, UniqueKeyLoader
from skills.registry import default_registry


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    memory_access: str = "none"

    @model_validator(mode="after")
    def no_allow_block_overlap(self) -> AgentConfig:
        overlap = set(self.allowed_tools) & set(self.blocked_tools)
        if overlap:
            raise ValueError(f"agent allows and blocks the same tools: {sorted(overlap)}")
        return self


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: dict[str, AgentConfig]


class CommandAllowlistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executable: str
    subcommands: list[str] = Field(default_factory=list)
    permission_level: int = Field(ge=0, le=5)
    risk_level: str


class ToolPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_external_actions: bool = False
    command_allowlist: list[CommandAllowlistItem] = Field(default_factory=list)


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: ToolPolicyConfig


class PermissionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    levels: dict[int, str]
    approval_required_at_level: int = Field(ge=0, le=5)
    external_actions_enabled: bool = False


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
    tools = default_registry()
    tool_names = {tool.name for tool in tools.list()}
    agent_names = {agent.name for agent in default_agent_registry().list()}
    agents_data = _read_yaml(root / "configs" / "agents.yaml", errors)
    agents_config: AgentsConfig | None = None
    if agents_data is not None:
        try:
            agents_config = AgentsConfig.model_validate(agents_data)
        except ValidationError as exc:
            errors.append(f"configs/agents.yaml: {exc}")
    if agents_config is not None:
        for name, agent in agents_config.agents.items():
            if name not in agent_names:
                errors.append(f"configs/agents.yaml: unknown agent reference: {name}")
            if (
                agent.model_id
                and model_registry is not None
                and not model_registry.exists(agent.model_id)
            ):
                errors.append(
                    f"configs/agents.yaml: agent {name} references unknown model {agent.model_id}"
                )
            for tool in [*agent.allowed_tools, *agent.blocked_tools]:
                if tool not in tool_names:
                    errors.append(
                        f"configs/agents.yaml: agent {name} references unknown tool {tool}"
                    )
    tools_data = _read_yaml(root / "configs" / "tools.yaml", errors)
    if tools_data is not None:
        try:
            ToolsConfig.model_validate(tools_data)
        except ValidationError as exc:
            errors.append(f"configs/tools.yaml: {exc}")
    permissions_data = _read_yaml(root / "configs" / "permissions.yaml", errors)
    if permissions_data is not None:
        try:
            PermissionsConfig.model_validate(permissions_data)
        except ValidationError as exc:
            errors.append(f"configs/permissions.yaml: {exc}")
    if settings is not None and settings.api.host != "127.0.0.1":
        errors.append("configs/april.yaml: API host should default to 127.0.0.1")
    if settings is not None and settings.runtime.host != "127.0.0.1":
        errors.append("configs/april.yaml: Runtime host should default to 127.0.0.1")
    return errors


def _read_yaml(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader) or {}
    except (OSError, yaml.YAMLError, ConfigError) as exc:
        errors.append(f"{path.name}: {exc}")
        return None
    if not isinstance(loaded, dict):
        errors.append(f"{path.name}: top-level document must be a mapping")
        return None
    return loaded
