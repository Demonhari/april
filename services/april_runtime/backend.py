from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.schemas import ChatMessage, FinishReason, ResponseFormat


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
    supports_native_batch_embeddings: bool = False
    # A candidate is safe only when every instance owns its backend object.  The
    # built-in backends satisfy this contract; lifecycle code also checks object
    # identity at load time so an accidental singleton factory fails closed.
    supports_isolated_instances: bool = True

    @abstractmethod
    async def load(self, model: ModelDefinition) -> None:
        raise NotImplementedError  # pragma: no cover - abstract contract

    @abstractmethod
    async def unload(self) -> None:
        raise NotImplementedError  # pragma: no cover - abstract contract

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
        raise NotImplementedError  # pragma: no cover - abstract contract

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
        raise NotImplementedError  # pragma: no cover - abstract contract

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
        response_format: ResponseFormat | None = None,
        disable_thinking: bool = False,
        prompt_tokens: int | None = None,
    ) -> GenerationResult:
        # Backends that only implement prompt completion ignore response_format and
        # rely on prompt-plus-validation; chat-capable backends override this.
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
        response_format: ResponseFormat | None = None,
        disable_thinking: bool = False,
        prompt_tokens: int | None = None,
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
        raise NotImplementedError  # pragma: no cover - abstract contract

    async def count_tokens(self, text: str) -> int:
        return len(await self.tokenize(text))

    def prompt_metadata(self) -> dict[str, object]:
        """Backend-provided metadata consulted by the prompt renderer.

        Returns an empty mapping by default. Backends that can read GGUF/native
        tokenizer metadata (currently only :class:`LlamaCppBackend`) override this
        to expose *only* the keys the renderer needs (native chat template /
        chat format). Raw template text is never logged, reported, or surfaced in
        health/readiness output — it is used solely for in-process rendering.
        """
        return {}

    async def embed(self, text: str) -> list[float]:
        raise RuntimeUnavailableError("backend does not support embeddings")

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Bounded sequential compatibility fallback; never runs concurrently."""
        return [await self.embed(text) for text in texts]

    def apply_thread_budget(self, n_threads: int, n_threads_batch: int) -> bool:
        del n_threads, n_threads_batch
        return False

    def timing_diagnostics(self) -> dict[str, object]:
        return {}

    def finish_timing_diagnostics(self, prompt_tokens: int, output_tokens: int) -> None:
        """Finalize optional native timing counters after a streamed turn."""
        del prompt_tokens, output_tokens

    def prefix_cache_diagnostics(self) -> dict[str, object]:
        return {}

    def prefix_cache_aggregate_diagnostics(self) -> dict[str, object]:
        return {}

    @abstractmethod
    async def health(self) -> BackendHealth:
        raise NotImplementedError  # pragma: no cover - abstract contract
