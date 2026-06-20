from __future__ import annotations

from pathlib import Path

from agents.schemas import AgentConfig


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


class BaseAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model_id(self) -> str | None:
        return self.config.model_id

    @property
    def system_prompt(self) -> str:
        return self.config.system_prompt
