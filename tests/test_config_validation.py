from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from april_common.config_validation import validate_configuration


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
