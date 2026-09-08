from __future__ import annotations

from pathlib import Path

from agents.schemas import AgentConfig

USER_FACING_ASSISTANT_NAME = "APRIL"
USER_FACING_IDENTITY_RULE = (
    "User-facing assistant identity: APRIL. Internal agent names and call signs are "
    "implementation metadata only. Answer interactive users as APRIL; do not introduce "
    "yourself as an internal call sign unless the user explicitly asks about APRIL's "
    "internal agent architecture."
)


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
