from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, cast

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.backend import BackendHealth, GenerationResult, RuntimeBackend
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.prompt_templates import (
    CHAT_FORMAT_METADATA_KEYS,
    NATIVE_TEMPLATE_METADATA_KEYS,
)
from services.april_runtime.schemas import ChatMessage, FinishReason, ResponseFormat
from services.april_runtime.stream_pump import pump_token_stream


def llama_response_format(response_format: ResponseFormat | None) -> dict[str, Any] | None:
    """Translate an APRIL ResponseFormat into llama-cpp-python's response_format.

    llama-cpp-python expects ``{"type": "json_object", "schema": <json schema>}``
    (the schema is optional). Returns None when no JSON constraint was requested.
    """
    if response_format is None or response_format.type == "text":
        return None
    payload: dict[str, Any] = {"type": "json_object"}
    if response_format.json_schema is not None:
        payload["schema"] = response_format.json_schema
    return payload


class LlamaCppBackend(RuntimeBackend):
    supports_concurrent_generation = False

    def __init__(self) -> None:
        self._llm: Any | None = None
        self._model: ModelDefinition | None = None
        self.last_prompt_path: str | None = None
        # Only the prompt-rendering keys the renderer consults are retained, never
        # the raw template for logging/reporting. Populated after a successful load.
        self._prompt_metadata: dict[str, object] = {}

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
        if model.role == "embedding":
            # A chat Llama instance cannot also embed; an embedding-role model is
            # loaded as its own dedicated instance with embedding mode enabled.
            kwargs["embedding"] = True
        self._llm = await asyncio.to_thread(Llama, **kwargs)
        self._prompt_metadata = _extract_prompt_metadata(self._llm)

    def prompt_metadata(self) -> dict[str, object]:
        return dict(self._prompt_metadata)

    async def unload(self) -> None:
        llm = self._llm
        self._llm = None
        self._model = None
        self._prompt_metadata = {}
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
    ) -> GenerationResult:
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        chat_completion = getattr(self._llm, "create_chat_completion", None)
        if not callable(chat_completion):
            self.last_prompt_path = "fallback_prompt"
            return await self.generate(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop=stop,
                seed=seed,
            )

        format_kwarg = llama_response_format(response_format)

        def run() -> Any:
            extra: dict[str, Any] = {}
            if format_kwarg is not None:
                extra["response_format"] = format_kwarg
            return chat_completion(
                messages=self._message_dicts(messages),
                **self._completion_kwargs(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    stream=False,
                    top_p=top_p,
                    stop=stop,
                    seed=seed,
                ),
                **extra,
            )

        try:
            output = await asyncio.to_thread(run)
        except Exception:
            # A backend/model that cannot honour response_format (or chat at all)
            # degrades to prompt completion plus downstream validation.
            self.last_prompt_path = "fallback_prompt"
            return await self.generate(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop=stop,
                seed=seed,
            )
        self.last_prompt_path = "chat_template"
        return await self._chat_generation_result(output, prompt)

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

        def make_iterator(is_cancelled: Callable[[], bool]) -> Iterator[str]:
            kwargs = self._completion_kwargs(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                stream=True,
                top_p=top_p,
                stop=stop,
                seed=seed,
            )
            self._add_stopping_criteria(kwargs, is_cancelled)
            for chunk in llm(prompt, **kwargs):
                if is_cancelled():
                    return
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield str(text)

        async for token in pump_token_stream(make_iterator):
            yield token

    async def stream_messages(
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
    ) -> AsyncIterator[str]:
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        chat_completion = getattr(self._llm, "create_chat_completion", None)
        if not callable(chat_completion):
            self.last_prompt_path = "fallback_prompt"
            async for token in self.stream(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop=stop,
                seed=seed,
            ):
                yield token
            return

        llm = self._llm
        format_kwarg = llama_response_format(response_format)

        def make_iterator(is_cancelled: Callable[[], bool]) -> Iterator[str]:
            chat_kwargs = self._completion_kwargs(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                stream=True,
                top_p=top_p,
                stop=stop,
                seed=seed,
            )
            if format_kwarg is not None:
                chat_kwargs["response_format"] = format_kwarg
            self._add_stopping_criteria(chat_kwargs, is_cancelled)
            emitted = False
            try:
                for chunk in chat_completion(
                    messages=self._message_dicts(messages),
                    **chat_kwargs,
                ):
                    if is_cancelled():
                        return
                    text = self._chat_stream_text(chunk)
                    if text:
                        emitted = True
                        self.last_prompt_path = "chat_template"
                        yield text
            except Exception:
                # If nothing was emitted yet, degrade to prompt completion; once
                # tokens are flowing a mid-stream failure is surfaced to the caller.
                if emitted:
                    raise
                prompt_kwargs = self._completion_kwargs(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    stream=True,
                    top_p=top_p,
                    stop=stop,
                    seed=seed,
                )
                self._add_stopping_criteria(prompt_kwargs, is_cancelled)
                for chunk in llm(prompt, **prompt_kwargs):
                    if is_cancelled():
                        return
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        self.last_prompt_path = "fallback_prompt"
                        yield str(text)

        async for token in pump_token_stream(make_iterator):
            yield token

    def _add_stopping_criteria(
        self, kwargs: dict[str, Any], is_cancelled: Callable[[], bool]
    ) -> None:
        # Wire llama.cpp's per-token stopping hook so cancellation is observed
        # mid-generation where the build supports it. Absence of the symbol is a
        # safe fallback: the pump's between-token checks still bound generation.
        try:
            from llama_cpp import StoppingCriteriaList
        except Exception:
            return

        def _criteria(input_ids: Any, logits: Any) -> bool:
            return is_cancelled()

        kwargs["stopping_criteria"] = StoppingCriteriaList([_criteria])

    async def tokenize(self, text: str) -> list[int]:
        if self._llm is None:
            return [index for index, _ in enumerate(text.split())]
        return list(await asyncio.to_thread(self._llm.tokenize, text.encode("utf-8")))

    async def embed(self, text: str) -> list[float]:
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        if self._model is not None and self._model.role != "embedding":
            raise RuntimeUnavailableError(
                "Loaded model is not an embedding model; load a role=embedding model to embed."
            )
        embedder = getattr(self._llm, "embed", None)
        if not callable(embedder):
            raise RuntimeUnavailableError("backend does not support embeddings")
        raw = await asyncio.to_thread(embedder, text)
        return _flatten_embedding(raw)

    async def health(self) -> BackendHealth:
        if self._llm is None:
            return BackendHealth(ok=False, message="not loaded")
        return BackendHealth(ok=True, message="loaded")

    def _completion_kwargs(
        self,
        *,
        max_output_tokens: int,
        temperature: float,
        stream: bool,
        top_p: float | None,
        stop: list[str] | None,
        seed: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop:
            kwargs["stop"] = stop
        if seed is not None:
            kwargs["seed"] = seed
        return kwargs

    def _message_dicts(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    async def _chat_generation_result(self, output: Any, prompt: str) -> GenerationResult:
        choice = output["choices"][0]
        message = choice.get("message") or {}
        text = str(message.get("content") or choice.get("text") or "")
        usage = output.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or await self.count_tokens(prompt))
        output_tokens = int(usage.get("completion_tokens") or await self.count_tokens(text))
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=self._finish_reason(choice.get("finish_reason")),
        )

    def _chat_stream_text(self, chunk: Any) -> str:
        choice = chunk["choices"][0]
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        return str(delta.get("content") or message.get("content") or choice.get("text") or "")

    def _finish_reason(self, raw: object) -> FinishReason:
        if raw in {"stop", "length", "error", "cancelled"}:
            return cast(FinishReason, raw)
        return "stop"


def _extract_prompt_metadata(llm: Any) -> dict[str, object]:
    """Safely read prompt-rendering metadata from a loaded llama-cpp ``Llama``.

    Only the stable GGUF/native tokenizer keys the renderer consults are kept
    (native chat template and, if genuinely present, a chat-format hint). Every
    access is a defensive ``getattr``/``dict.get`` so a llama-cpp-python version
    without a ``metadata`` mapping simply yields ``{}`` and the renderer falls
    back to explicit config / name inference. The raw chat template is retained
    only for in-process rendering — it is never logged, reported, or surfaced.
    """
    metadata: dict[str, object] = {}
    raw = getattr(llm, "metadata", None)
    if isinstance(raw, dict):
        for key in (*NATIVE_TEMPLATE_METADATA_KEYS, *CHAT_FORMAT_METADATA_KEYS):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value
    return metadata


def _flatten_embedding(raw: Any) -> list[float]:
    # llama-cpp-python may return a flat vector or a list of per-token vectors.
    values = list(raw)
    if values and isinstance(values[0], (list, tuple)):
        return [float(component) for component in values[0]]
    return [float(component) for component in values]
