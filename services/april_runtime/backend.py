from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.schemas import FinishReason


@dataclass(frozen=True, slots=True)
class BackendHealth:
    ok: bool
    message: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: FinishReason = "stop"


class RuntimeBackend(ABC):
    supports_concurrent_generation: bool = False

    @abstractmethod
    async def load(self, model: ModelDefinition) -> None:
        raise NotImplementedError

    @abstractmethod
    async def unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    async def tokenize(self, text: str) -> list[int]:
        raise NotImplementedError

    async def count_tokens(self, text: str) -> int:
        return len(await self.tokenize(text))

    @abstractmethod
    async def health(self) -> BackendHealth:
        raise NotImplementedError
