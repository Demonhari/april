from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.backend import BackendHealth, GenerationResult, RuntimeBackend
from services.april_runtime.model_registry import ModelDefinition


class LlamaCppBackend(RuntimeBackend):
    supports_concurrent_generation = False

    def __init__(self) -> None:
        self._llm: Any | None = None
        self._model: ModelDefinition | None = None

    async def load(self, model: ModelDefinition) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "Optional dependency llama-cpp-python is not installed. "
                "Install with `pip install .[runtime]` or set APRIL_RUNTIME_BACKEND=fake.",
                {"model_id": model.id},
            ) from exc
        path = model.path.expanduser().resolve(strict=False)
        if not path.exists():
            raise RuntimeUnavailableError(
                "Configured GGUF model file is missing.", {"path": str(path)}
            )
        self._model = model
        self._llm = await asyncio.to_thread(
            Llama,
            model_path=str(path),
            n_ctx=model.context_size,
            n_threads=model.threads,
            verbose=False,
        )

    async def unload(self) -> None:
        self._llm = None
        self._model = None
        await asyncio.sleep(0)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> GenerationResult:
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        llm = self._llm

        def run() -> Any:
            return llm(
                prompt,
                max_tokens=max_output_tokens,
                temperature=temperature,
                stream=False,
            )

        output = await asyncio.to_thread(run)
        choice = output["choices"][0]
        text = str(choice.get("text", ""))
        input_tokens = await self.count_tokens(prompt)
        output_tokens = await self.count_tokens(text)
        return GenerationResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens)

    async def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> AsyncIterator[str]:
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        llm = self._llm

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def produce() -> None:
            try:
                for chunk in llm(
                    prompt,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    stream=True,
                ):
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        queue.put_nowait(str(text))
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                token = await queue.get()
                if token is None:
                    break
                yield token
        finally:
            await task

    async def tokenize(self, text: str) -> list[int]:
        if self._llm is None:
            return [index for index, _ in enumerate(text.split())]
        return list(await asyncio.to_thread(self._llm.tokenize, text.encode("utf-8")))

    async def health(self) -> BackendHealth:
        if self._llm is None:
            return BackendHealth(ok=False, message="not loaded")
        return BackendHealth(ok=True, message="loaded")
