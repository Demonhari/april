from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.schemas import ChatMessage, FinishReason


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
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    async def generate_messages(
        self,
        prompt: str,
        *,
        messages: list[ChatMessage],
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        return await self.generate(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
        )

    def stream_messages(
        self,
        prompt: str,
        *,
        messages: list[ChatMessage],
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        return self.stream(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
        )

    @abstractmethod
    async def tokenize(self, text: str) -> list[int]:
        raise NotImplementedError

    async def count_tokens(self, text: str) -> int:
        return len(await self.tokenize(text))

    async def embed(self, text: str) -> list[float]:
        raise RuntimeUnavailableError("backend does not support embeddings")

    @abstractmethod
    async def health(self) -> BackendHealth:
        raise NotImplementedError
