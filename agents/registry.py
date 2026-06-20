from __future__ import annotations

from agents.base import BaseAgent
from agents.coding.agent import coding_agent
from agents.creative.agent import creative_agent
from agents.general.agent import general_agent
from agents.reading.agent import reading_agent
from agents.reasoning.agent import reasoning_agent
from agents.system_action.agent import system_action_agent


class AgentRegistry:
    def __init__(self, agents: list[BaseAgent]) -> None:
        self._agents = {agent.name: agent for agent in agents}

    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    def list(self) -> list[BaseAgent]:
        return list(self._agents.values())


def default_agent_registry() -> AgentRegistry:
    return AgentRegistry(
        [
            general_agent(),
            coding_agent(),
            reading_agent(),
            creative_agent(),
            reasoning_agent(),
            system_action_agent(),
        ]
    )
