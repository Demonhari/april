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
        kwargs: dict[str, Any] = {
            "model_path": str(path),
            "n_ctx": model.context_size,
            "n_threads": model.threads,
            "verbose": False,
        }
        optional_values = {
            "n_gpu_layers": model.n_gpu_layers,
            "n_batch": model.n_batch,
            "n_ubatch": model.n_ubatch,
            "use_mmap": model.use_mmap,
            "use_mlock": model.use_mlock,
            "chat_format": model.chat_format,
        }
        kwargs.update({key: value for key, value in optional_values.items() if value is not None})
        self._llm = await asyncio.to_thread(Llama, **kwargs)

    async def unload(self) -> None:
        llm = self._llm
        self._llm = None
        self._model = None
        if llm is not None:
            close = getattr(llm, "close", None) or getattr(llm, "release", None)
            if callable(close):
                await asyncio.to_thread(close)
        await asyncio.sleep(0)

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
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        llm = self._llm

        def run() -> Any:
            kwargs: dict[str, Any] = {
                "max_tokens": max_output_tokens,
                "temperature": temperature,
                "stream": False,
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            if stop:
                kwargs["stop"] = stop
            if seed is not None:
                kwargs["seed"] = seed
            return llm(prompt, **kwargs)

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
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        llm = self._llm

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue(maxsize=32)

        def produce() -> None:
            def put(item: str | Exception | None) -> None:
                asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()

            try:
                kwargs: dict[str, Any] = {
                    "max_tokens": max_output_tokens,
                    "temperature": temperature,
                    "stream": True,
                }
                if top_p is not None:
                    kwargs["top_p"] = top_p
                if stop:
                    kwargs["stop"] = stop
                if seed is not None:
                    kwargs["seed"] = seed
                for chunk in llm(
                    prompt,
                    **kwargs,
                ):
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        put(str(text))
            except Exception as exc:
                put(exc)
            finally:
                put(None)

        task = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                token = await queue.get()
                if token is None:
                    break
                if isinstance(token, Exception):
                    raise token
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
