from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agents.base import BaseAgent
from agents.registry import AgentRegistry
from agents.schemas import AgentConfig as RuntimeAgentConfig
from april_common.errors import ConfigError
from april_common.settings import project_root
from services.april_runtime.model_registry import ModelRegistry
from skills.registry import ToolRegistry, default_registry

KNOWN_AGENT_IDS = {
    "coding_agent",
    "creative_agent",
    "general_agent",
    "reading_agent",
    "reasoning_agent",
    "system_action_agent",
}
MEMORY_POLICIES = {"none", "conversation_and_safe_memory", "project_memory"}
RISK_LEVELS = {
    "none",
    "read_only",
    "safe_write",
    "code_write",
    "system_action",
    "external_action",
}
DENIED_COMMAND_EXECUTABLES = {
    "bash",
    "brew",
    "chmod",
    "chown",
    "conda",
    "curl",
    "fish",
    "mv",
    "npm",
    "pip",
    "pip3",
    "pnpm",
    "rm",
    "sh",
    "yarn",
    "zsh",
}
ALLOWED_PYTHON_MODULES = {"timeit", "pytest", "ruff"}


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> Any:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"Duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


class ConfiguredAgent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    model_id: str | None = None
    prompt_path: str
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    memory_access: Literal["none", "conversation_and_safe_memory", "project_memory"] = "none"
    maximum_tool_iterations: int = Field(default=5, ge=1, le=20)
    output_schema: str = "AgentResult"

    @model_validator(mode="after")
    def no_allow_block_overlap(self) -> ConfiguredAgent:
        overlap = set(self.allowed_tools) & set(self.blocked_tools)
        if overlap:
            raise ValueError(f"agent allows and blocks the same tools: {sorted(overlap)}")
        return self


class AgentsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: dict[str, ConfiguredAgent]


class CommandAllowlistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executable: str
    subcommands: list[str] = Field(default_factory=list)
    permission_level: int = Field(ge=0, le=5)
    risk_level: str

    @field_validator("executable")
    @classmethod
    def executable_is_plain_name(cls, value: str) -> str:
        if Path(value).name != value:
            raise ValueError("command executable must be a plain name, not a path")
        if value in DENIED_COMMAND_EXECUTABLES:
            raise ValueError(f"command executable is permanently denied: {value}")
        return value

    @field_validator("risk_level")
    @classmethod
    def risk_is_known(cls, value: str) -> str:
        if value not in RISK_LEVELS:
            raise ValueError(f"unknown risk level: {value}")
        return value

    @model_validator(mode="after")
    def python_must_remain_constrained(self) -> CommandAllowlistItem:
        if self.executable == "python" and self.subcommands != ["-m"]:
            raise ValueError("python command rules may only allow the -m subcommand")
        return self


def _default_url_schemes() -> list[Literal["http", "https"]]:
    return ["http", "https"]


class ToolPolicyFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_allowlist: list[CommandAllowlistItem] = Field(default_factory=list)
    open_app_allowlist: list[str] = Field(default_factory=list)
    open_url_allowed_schemes: list[Literal["http", "https"]] = Field(
        default_factory=lambda: _default_url_schemes()
    )

    @field_validator("open_app_allowlist")
    @classmethod
    def app_names_are_plain(cls, value: list[str]) -> list[str]:
        for name in value:
            if not name.strip() or "/" in name or "\x00" in name:
                raise ValueError("open_app allowlist entries must be plain application names")
        return value


class ToolsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: ToolPolicyFile


class PermissionsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    levels: dict[int, str]
    approval_required_at_level: int = Field(ge=0, le=5)
    external_actions_enabled: bool = False

    @model_validator(mode="after")
    def cannot_weaken_level_three_approval(self) -> PermissionsFile:
        if self.approval_required_at_level > 3:
            raise ValueError("Level 3 and above must require approval")
        for level in range(0, 6):
            if level not in self.levels:
                raise ValueError(f"missing permission level {level}")
        return self


def load_agents_file(home: Path) -> AgentsFile:
    data = _read_yaml(home / "configs" / "agents.yaml", default={"agents": {}})
    try:
        return AgentsFile.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid agents.yaml: {_validation_summary(exc)}",
            {"errors": exc.errors()},
        ) from exc


def load_tools_file(home: Path) -> ToolsFile:
    data = _read_yaml(
        home / "configs" / "tools.yaml",
        default={
            "tools": {
                "command_allowlist": [
                    {
                        "executable": "pytest",
                        "subcommands": [],
                        "permission_level": 3,
                        "risk_level": "code_write",
                    },
                    {
                        "executable": "ruff",
                        "subcommands": ["check", "format"],
                        "permission_level": 3,
                        "risk_level": "code_write",
                    },
                    {
                        "executable": "python",
                        "subcommands": ["-m"],
                        "permission_level": 3,
                        "risk_level": "code_write",
                    },
                ],
                "open_app_allowlist": [],
                "open_url_allowed_schemes": ["http", "https"],
            }
        },
    )
    try:
        return ToolsFile.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid tools.yaml: {_validation_summary(exc)}",
            {"errors": exc.errors()},
        ) from exc


def load_permissions_file(home: Path) -> PermissionsFile:
    data = _read_yaml(
        home / "configs" / "permissions.yaml",
        default={
            "levels": {
                0: "no_tools",
                1: "read_only",
                2: "safe_local_write",
                3: "code_write",
                4: "system_action",
                5: "external_action",
            },
            "approval_required_at_level": 3,
            "external_actions_enabled": False,
        },
    )
    try:
        return PermissionsFile.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid permissions.yaml: {_validation_summary(exc)}",
            {"errors": exc.errors()},
        ) from exc


def build_agent_registry_from_config(
    *,
    home: Path,
    model_registry: ModelRegistry,
    tool_registry: ToolRegistry,
) -> AgentRegistry:
    config = load_agents_file(home)
    tool_names = {definition.name for definition in tool_registry.list()}
    agents: list[BaseAgent] = []
    seen: set[str] = set()
    for agent_id, agent in config.agents.items():
        if agent_id in seen:
            raise ConfigError(f"Duplicate agent ID: {agent_id}")
        seen.add(agent_id)
        if agent_id not in KNOWN_AGENT_IDS:
            raise ConfigError(f"Unknown agent ID in agents.yaml: {agent_id}")
        if agent.model_id is not None and not model_registry.exists(agent.model_id):
            raise ConfigError(f"Agent {agent_id} references unknown model: {agent.model_id}")
        for tool_name in [*agent.allowed_tools, *agent.blocked_tools]:
            if tool_name not in tool_names:
                raise ConfigError(f"Agent {agent_id} references unknown tool: {tool_name}")
        prompt_path = _resolve_prompt_path(home, agent.prompt_path)
        if not prompt_path.exists():
            raise ConfigError(f"Agent {agent_id} prompt path does not exist: {prompt_path}")
        agents.append(
            BaseAgent(
                RuntimeAgentConfig(
                    name=agent_id,
                    description=agent.description,
                    model_id=agent.model_id,
                    system_prompt_path=str(prompt_path),
                    allowed_tools=set(agent.allowed_tools),
                    blocked_tools=set(agent.blocked_tools),
                    memory_access_policy=agent.memory_access,
                    maximum_tool_iterations=agent.maximum_tool_iterations,
                    output_schema=agent.output_schema,
                    system_prompt=prompt_path.read_text(encoding="utf-8").strip(),
                )
            )
        )
    missing = KNOWN_AGENT_IDS - seen
    if missing:
        raise ConfigError(f"agents.yaml is missing required agents: {sorted(missing)}")
    return AgentRegistry(agents)


def build_configured_tool_registry(home: Path, agents: AgentRegistry) -> ToolRegistry:
    load_tools_file(home)
    registry = default_registry()
    allowed_by_tool: dict[str, set[str]] = {
        definition.name: set() for definition in registry.list()
    }
    for agent in agents.list():
        for tool in agent.config.allowed_tools:
            if tool not in allowed_by_tool:
                raise ConfigError(f"Agent {agent.name} references unknown tool: {tool}")
            allowed_by_tool[tool].add(agent.name)
    for definition in registry.list():
        definition.allowed_agents = allowed_by_tool.get(definition.name, set())
    return registry


def _resolve_prompt_path(home: Path, raw_path: str) -> Path:
    configured = (home / raw_path).expanduser().resolve()
    if configured.exists():
        return configured
    bundled = (project_root() / raw_path).expanduser().resolve()
    if bundled.exists():
        return bundled
    return configured


def _read_yaml(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader) or {}
    except (OSError, yaml.YAMLError, ConfigError) as exc:
        raise ConfigError(f"Failed to read {path.name}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path.name}: top-level document must be a mapping")
    return loaded


def _validation_summary(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {error.get('msg', 'invalid value')}")
    return "; ".join(parts)
