from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from april_common.config_validation import validate_configuration
from april_common.effective_config import (
    build_agent_registry_from_config,
    build_configured_tool_registry,
    load_permissions_file,
)
from services.april_runtime.model_registry import ModelRegistry
from skills.registry import default_registry


def copy_configs(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    shutil.copytree(Path.cwd() / "configs", home / "configs")
    return home


def test_config_validation_accepts_default_configs(tmp_path: Path) -> None:
    home = copy_configs(tmp_path)
    assert validate_configuration(home) == []


def test_config_validation_rejects_unknown_agent_tool(tmp_path: Path) -> None:
    home = copy_configs(tmp_path)
    agents_path = home / "configs" / "agents.yaml"
    data = yaml.safe_load(agents_path.read_text(encoding="utf-8"))
    data["agents"]["coding_agent"]["allowed_tools"].append("missing_tool")
    agents_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    errors = validate_configuration(home)
    assert any("missing_tool" in error for error in errors)


def test_config_validation_rejects_unknown_field(tmp_path: Path) -> None:
    home = copy_configs(tmp_path)
    tools_path = home / "configs" / "tools.yaml"
    data = yaml.safe_load(tools_path.read_text(encoding="utf-8"))
    data["tools"]["allow_external_actions"] = False
    tools_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    errors = validate_configuration(home)
    assert any("allow_external_actions" in error for error in errors)


def test_config_validation_rejects_unsafe_command_allowlist(tmp_path: Path) -> None:
    home = copy_configs(tmp_path)
    tools_path = home / "configs" / "tools.yaml"
    data = yaml.safe_load(tools_path.read_text(encoding="utf-8"))
    data["tools"]["command_allowlist"].append(
        {
            "executable": "bash",
            "subcommands": ["-c"],
            "permission_level": 3,
            "risk_level": "code_write",
        }
    )
    tools_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    errors = validate_configuration(home)
    assert any("permanently denied" in error for error in errors)


def test_agent_config_changes_effective_runtime_registry(tmp_path: Path) -> None:
    home = copy_configs(tmp_path)
    agents_path = home / "configs" / "agents.yaml"
    data = yaml.safe_load(agents_path.read_text(encoding="utf-8"))
    data["agents"]["coding_agent"]["maximum_tool_iterations"] = 2
    data["agents"]["coding_agent"]["allowed_tools"] = ["read_file"]
    agents_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    model_registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    agent_registry = build_agent_registry_from_config(
        home=home,
        model_registry=model_registry,
        tool_registry=default_registry(),
    )
    coding = agent_registry.get("coding_agent")
    assert coding is not None
    assert coding.config.maximum_tool_iterations == 2
    assert coding.config.allowed_tools == {"read_file"}

    tool_registry = build_configured_tool_registry(home, agent_registry)
    assert tool_registry.get("read_file") is not None
    assert tool_registry.get("read_file").allowed_agents == {  # type: ignore[union-attr]
        "coding_agent",
        "reading_agent",
        "reasoning_agent",
    }
    assert tool_registry.get("patch_applier").allowed_agents == set()  # type: ignore[union-attr]


def test_permissions_yaml_controls_approval_threshold(tmp_path: Path) -> None:
    home = copy_configs(tmp_path)
    permissions_path = home / "configs" / "permissions.yaml"
    data = yaml.safe_load(permissions_path.read_text(encoding="utf-8"))
    data["approval_required_at_level"] = 2
    data["external_actions_enabled"] = True
    permissions_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_permissions_file(home)
    assert config.approval_required_at_level == 2
    assert config.external_actions_enabled is True


def test_permissions_yaml_cannot_weaken_level_three_approval(tmp_path: Path) -> None:
    home = copy_configs(tmp_path)
    permissions_path = home / "configs" / "permissions.yaml"
    data = yaml.safe_load(permissions_path.read_text(encoding="utf-8"))
    data["approval_required_at_level"] = 4
    permissions_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    errors = validate_configuration(home)
    assert any("Level 3" in error for error in errors)
