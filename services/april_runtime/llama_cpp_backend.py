from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, cast

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.backend import BackendHealth, GenerationResult, RuntimeBackend
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.prefix_cache import PrefixStatePolicy
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


def llama_chat_format(chat_format: str | None) -> str | None:
    """Return a llama-cpp-python chat handler name for APRIL's prompt family.

    APRIL's ``chat_format`` is primarily a prompt-rendering family understood by
    :mod:`services.april_runtime.prompt_templates`. llama-cpp-python has its own
    narrower set of built-in chat handlers; passing an APRIL-only family such as
    ``granite`` makes real strict chat fail at runtime. Only pass through handler
    names that are known to be native llama-cpp-python handlers. Other APRIL
    formats still render through APRIL templates and may use GGUF native chat
    template metadata inside llama-cpp-python.
    """
    if chat_format == "qwen":
        return "qwen"
    return None


class LlamaCppBackend(RuntimeBackend):
    supports_concurrent_generation = False

    def __init__(self, *, prefix_cache_enabled: bool = True) -> None:
        self._llm: Any | None = None
        self._model: ModelDefinition | None = None
        self.last_prompt_path: str | None = None
        self.last_structured_output_fallback = False
        self.last_structured_output_fallback_reason: str | None = None
        # Only the prompt-rendering keys the renderer consults are retained, never
        # the raw template for logging/reporting. Populated after a successful load.
        self._prompt_metadata: dict[str, object] = {}
        self._prefix_policy: PrefixStatePolicy | None = None
        self._prefix_adapter: Any | None = None
        self._prefix_disabled_reason: str | None = None
        self._prefix_lookup_hit = False
        self._prefix_lookup_diagnostics: dict[str, object] = {}
        self._last_timing: dict[str, object] = {}
        self._perf_started = False
        self._llama_module: Any | None = None
        self._prefix_cache_enabled = prefix_cache_enabled

    async def load(self, model: ModelDefinition) -> None:
        try:
            import llama_cpp

            Llama = llama_cpp.Llama
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "Optional dependency llama-cpp-python is not installed. "
                "Install with `pip install .[runtime]` or set APRIL_RUNTIME_BACKEND=fake.",
                {"model_id": model.id},
            ) from exc
        self._llama_module = llama_cpp
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
            "n_threads_batch": model.threads_batch or model.threads,
            "verbose": False,
        }
        optional_values = {
            "n_gpu_layers": model.n_gpu_layers,
            "n_batch": model.n_batch,
            "n_ubatch": model.n_ubatch,
            "use_mmap": model.use_mmap,
            "use_mlock": model.use_mlock,
            "chat_format": llama_chat_format(model.chat_format),
            "flash_attn": model.flash_attn,
        }
        kwargs.update({key: value for key, value in optional_values.items() if value is not None})
        if model.adapter_path is not None:
            adapter = model.adapter_path.expanduser().resolve(strict=False)
            if not adapter.exists():
                # Fail closed with an actionable error instead of silently
                # serving the base model without its configured adapter.
                raise RuntimeUnavailableError(
                    "Configured LoRA adapter file is missing.",
                    {"model_id": model.id, "adapter": adapter.name},
                )
            kwargs["lora_path"] = str(adapter)
        if model.role == "embedding":
            # A chat Llama instance cannot also embed; an embedding-role model is
            # loaded as its own dedicated instance with embedding mode enabled.
            kwargs["embedding"] = True
        self._llm = await asyncio.to_thread(Llama, **kwargs)
        self._prompt_metadata = _extract_prompt_metadata(self._llm)
        self._prefix_policy = None
        self._prefix_adapter = None
        self._prefix_disabled_reason = None
        if (
            model.role != "embedding"
            and (model.prefix_cache_mb or 0) > 0
            and self._prefix_cache_enabled
            and _prefix_cache_enabled()
        ):
            base_cache = getattr(llama_cpp, "BaseLlamaCache", None)
            if base_cache is not None and callable(getattr(self._llm, "set_cache", None)):
                self._prefix_policy = PrefixStatePolicy(
                    (model.prefix_cache_mb or 0) * 1024 * 1024,
                    model.prefix_cache_min_tokens,
                    64,
                )
                try:
                    self._prefix_adapter = _build_prefix_adapter(
                        base_cache,
                        self._llm,
                        self._prefix_policy,
                        lambda: self._disable_prefix_cache("adapter_error"),
                    )
                except Exception:
                    self._prefix_policy = None
                    self._prefix_adapter = None
                    self._prefix_disabled_reason = "cache_adapter_unavailable"
            else:
                self._prefix_disabled_reason = "cache_protocol_unavailable"
        elif not self._prefix_cache_enabled or not _prefix_cache_enabled():
            self._prefix_disabled_reason = "disabled_by_environment"

    def prompt_metadata(self) -> dict[str, object]:
        return dict(self._prompt_metadata)

    async def unload(self) -> None:
        llm = self._llm
        self._llm = None
        self._model = None
        self._prompt_metadata = {}
        self._prefix_policy = None
        self._prefix_adapter = None
        self._prefix_disabled_reason = None
        self._llama_module = None
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
        self._reset_generation_diagnostics()
        self.last_prompt_path = "prompt_completion"
        return await self._generate_prompt_completion(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
        )

    async def _generate_prompt_completion(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        prompt_tokens: int | None = None,
        _allow_cache_retry: bool = True,
    ) -> GenerationResult:
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        llm = self._llm
        if prompt_tokens is None:
            prompt_tokens = await self.count_tokens(prompt)
        self._attach_prefix_cache(prompt_tokens, prompt)
        self._begin_perf()

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

        try:
            output = await asyncio.to_thread(run)
        except Exception:
            self._refresh_prefix_lookup()
            if _allow_cache_retry and self._prefix_lookup_hit:
                self._recover_from_cache_failure()
                return await self._generate_prompt_completion(
                    prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_p=top_p,
                    stop=stop,
                    seed=seed,
                    prompt_tokens=prompt_tokens,
                    _allow_cache_retry=False,
                )
            raise
        self._refresh_prefix_lookup()
        choice = output["choices"][0]
        text = str(choice.get("text", ""))
        input_tokens = await self.count_tokens(prompt)
        output_tokens = await self.count_tokens(text)
        self._finish_perf(input_tokens, output_tokens)
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=self._finish_reason(choice.get("finish_reason")),
        )

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
        self._reset_generation_diagnostics()
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        if prompt_tokens is None:
            prompt_tokens = await self.count_tokens(prompt)
        chat_completion = getattr(self._llm, "create_chat_completion", None)
        if not callable(chat_completion):
            self._mark_prompt_fallback(
                response_format=response_format,
                reason="chat_completion_unavailable",
            )
            return await self._generate_prompt_completion(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop=stop,
                seed=seed,
                prompt_tokens=prompt_tokens,
            )

        format_kwarg = llama_response_format(response_format)

        def run() -> Any:
            extra: dict[str, Any] = {}
            if format_kwarg is not None:
                extra["response_format"] = format_kwarg
            return chat_completion(
                messages=self._message_dicts(messages, disable_thinking=disable_thinking),
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
            self._attach_prefix_cache(prompt_tokens, prompt)
            self._begin_perf()
            try:
                output = await asyncio.to_thread(run)
            except Exception:
                self._refresh_prefix_lookup()
                if self._prefix_lookup_hit:
                    self._recover_from_cache_failure()
                    output = await asyncio.to_thread(run)
                else:
                    raise
            self._refresh_prefix_lookup()
        except Exception as exc:
            if format_kwarg is not None and not _is_structured_fallback_exception(exc):
                raise
            # A backend/model that cannot honour response_format (or chat at all)
            # degrades to prompt completion plus downstream validation. Strict
            # callers see this through diagnostics so verification cannot mistake
            # prompt fallback for native structured/chat support.
            self._mark_prompt_fallback(
                response_format=response_format,
                reason=_fallback_reason(exc, response_format=response_format),
            )
            return await self._generate_prompt_completion(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop=stop,
                seed=seed,
                prompt_tokens=prompt_tokens,
            )
        self._mark_chat_success()
        result = await self._chat_generation_result(output, prompt)
        self._finish_perf(result.input_tokens, result.output_tokens)
        return result

    async def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        prompt_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self._reset_generation_diagnostics()
        self.last_prompt_path = "prompt_completion"
        async for token in self._stream_prompt_completion(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
            prompt_tokens=prompt_tokens,
        ):
            yield token

    async def _stream_prompt_completion(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        prompt_tokens: int | None = None,
        _allow_cache_retry: bool = True,
    ) -> AsyncIterator[str]:
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        llm = self._llm
        if prompt_tokens is None:
            prompt_tokens = await self.count_tokens(prompt)
        self._attach_prefix_cache(prompt_tokens, prompt)
        self._begin_perf()

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
            try:
                for chunk in llm(prompt, **kwargs):
                    if is_cancelled():
                        return
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        yield str(text)
            except Exception:
                self._refresh_prefix_lookup()
                raise

        try:
            async for token in pump_token_stream(make_iterator):
                yield token
        except Exception:
            self._refresh_prefix_lookup()
            if _allow_cache_retry and self._prefix_lookup_hit:
                self._recover_from_cache_failure()
                async for token in self._stream_prompt_completion(
                    prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_p=top_p,
                    stop=stop,
                    seed=seed,
                    prompt_tokens=prompt_tokens,
                    _allow_cache_retry=False,
                ):
                    yield token
                return
            raise

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
        disable_thinking: bool = False,
        prompt_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self._reset_generation_diagnostics()
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        chat_completion = getattr(self._llm, "create_chat_completion", None)
        if not callable(chat_completion):
            self._mark_prompt_fallback(
                response_format=response_format,
                reason="chat_completion_unavailable",
            )
            async for token in self._stream_prompt_completion(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop=stop,
                seed=seed,
                prompt_tokens=prompt_tokens,
            ):
                yield token
            return

        llm = self._llm
        if prompt_tokens is None:
            prompt_tokens = await self.count_tokens(prompt)
        self._attach_prefix_cache(prompt_tokens, prompt)
        self._begin_perf()
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
                    messages=self._message_dicts(messages, disable_thinking=disable_thinking),
                    **chat_kwargs,
                ):
                    if is_cancelled():
                        return
                    text = self._chat_stream_text(chunk)
                    if text:
                        emitted = True
                        self._mark_chat_success()
                        yield text
            except Exception as exc:
                # If nothing was emitted yet, degrade to prompt completion; once
                # tokens are flowing a mid-stream failure is surfaced to the caller.
                self._refresh_prefix_lookup()
                if not emitted and self._prefix_lookup_hit:
                    self._recover_from_cache_failure()
                    for chunk in chat_completion(
                        messages=self._message_dicts(messages, disable_thinking=disable_thinking),
                        **chat_kwargs,
                    ):
                        if is_cancelled():
                            return
                        text = self._chat_stream_text(chunk)
                        if text:
                            yield text
                    return
                if emitted:
                    raise
                if format_kwarg is not None and not _is_structured_fallback_exception(exc):
                    raise
                self._mark_prompt_fallback(
                    response_format=response_format,
                    reason=_fallback_reason(exc, response_format=response_format),
                )
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

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if self._llm is None:
            raise RuntimeUnavailableError("Model is not loaded.")
        if self._model is not None and self._model.role != "embedding":
            raise RuntimeUnavailableError(
                "Loaded model is not an embedding model; load a role=embedding model to embed."
            )
        embedder = getattr(self._llm, "embed", None)
        if not callable(embedder):
            raise RuntimeUnavailableError("backend does not support embeddings")
        try:
            raw = await asyncio.to_thread(embedder, texts)
        except (TypeError, AttributeError):
            self.supports_native_batch_embeddings = False
            return await super().embed_many(texts)
        vectors = _flatten_embedding_batch(raw)
        if len(vectors) != len(texts):
            raise ValueError("native embedding batch result count mismatch")
        self.supports_native_batch_embeddings = True
        return vectors

    async def health(self) -> BackendHealth:
        if self._llm is None:
            return BackendHealth(ok=False, message="not loaded")
        return BackendHealth(ok=True, message="loaded")

    def apply_thread_budget(self, n_threads: int, n_threads_batch: int) -> bool:
        llm = self._llm
        setter = getattr(self._llama_module, "llama_set_n_threads", None)
        ctx = getattr(llm, "ctx", None) if llm is not None else None
        if not callable(setter) or ctx is None:
            return False
        try:
            setter(ctx, n_threads, n_threads_batch)
        except Exception:
            return False
        return True

    def timing_diagnostics(self) -> dict[str, object]:
        return dict(self._last_timing)

    def finish_timing_diagnostics(self, prompt_tokens: int, output_tokens: int) -> None:
        self._finish_perf(prompt_tokens, output_tokens)

    def prefix_cache_diagnostics(self) -> dict[str, object]:
        result: dict[str, object] = {
            "enabled": self._prefix_policy is not None,
            "attached": self._prefix_adapter is not None
            and getattr(self._llm, "cache", None) is self._prefix_adapter,
            "disabled_reason": self._prefix_disabled_reason,
        }
        if self._prefix_policy is not None:
            result.update(self._prefix_policy.stats())
            result.update(self._prefix_lookup_diagnostics)
        if self._prefix_lookup_diagnostics.get("hit") is True:
            result["reused_prefix_tokens"] = max(
                int(cast(Any, self._prefix_lookup_diagnostics.get("live_prefix_tokens", 0))),
                int(cast(Any, self._prefix_lookup_diagnostics.get("cached_prefix_tokens", 0))),
            )
        return {key: value for key, value in result.items() if value is not None}

    def _reset_generation_diagnostics(self) -> None:
        self.last_prompt_path = None
        self.last_structured_output_fallback = False
        self.last_structured_output_fallback_reason = None
        self._prefix_lookup_hit = False
        self._prefix_lookup_diagnostics = {}
        self._last_timing = {}

    def _begin_perf(self) -> None:
        self._perf_started = False
        reset = getattr(self._llama_module, "llama_perf_context_reset", None)
        ctx = getattr(self._llm, "ctx", None)
        if callable(reset) and ctx is not None:
            try:
                reset(ctx)
                self._perf_started = True
            except Exception:
                pass

    def _refresh_prefix_lookup(self) -> None:
        if self._prefix_adapter is not None:
            self._prefix_lookup_diagnostics = dict(getattr(self._prefix_adapter, "last_lookup", {}))
            self._prefix_lookup_hit = bool(self._prefix_lookup_diagnostics.get("hit", False))

    def _finish_perf(self, prompt_tokens: int, output_tokens: int) -> None:
        self._last_timing = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "threads": self._model.threads if self._model else None,
            "threads_batch": (
                (self._model.threads_batch or self._model.threads) if self._model else None
            ),
        }
        if self._prefix_adapter is not None:
            self._prefix_lookup_diagnostics = dict(getattr(self._prefix_adapter, "last_lookup", {}))
        if not self._perf_started:
            self._last_timing = {
                key: value for key, value in self._last_timing.items() if value is not None
            }
            return
        reader = getattr(self._llama_module, "llama_perf_context", None)
        ctx = getattr(self._llm, "ctx", None)
        if not callable(reader) or ctx is None:
            return
        try:
            data = reader(ctx)
            values = {}
            for target, names in {
                "prompt_eval_tokens": ("n_p_eval", "prompt_eval_tokens"),
                "prompt_eval_ms": ("t_p_eval", "prompt_eval_ms"),
                "eval_tokens": ("n_eval", "eval_tokens"),
                "eval_ms": ("t_eval", "eval_ms"),
            }.items():
                value = next(
                    (data.get(name) for name in names if isinstance(data, dict) and name in data),
                    next((getattr(data, name) for name in names if hasattr(data, name)), None),
                )
                if value is not None:
                    values[target] = float(value) if target.endswith("_ms") else int(value)
            self._last_timing.update(values)
        except Exception:
            return

    def _attach_prefix_cache(self, prompt_tokens: int, prompt: str) -> None:
        if self._llm is None or self._prefix_adapter is None or self._model is None:
            return
        if prompt_tokens < self._model.prefix_cache_min_tokens:
            with contextlib.suppress(Exception):
                self._llm.set_cache(None)
            return
        try:
            self._llm.set_cache(self._prefix_adapter)
        except Exception:
            self._disable_prefix_cache("set_cache_failed")

    def _recover_from_cache_failure(self) -> None:
        llm = self._llm
        if llm is not None:
            reset = getattr(llm, "reset", None)
            if callable(reset):
                with contextlib.suppress(Exception):
                    reset()
        self._disable_prefix_cache("state_restore_failed")

    def _disable_prefix_cache(self, reason: str) -> None:
        self._prefix_disabled_reason = reason
        self._prefix_adapter = None
        self._prefix_policy = None
        if self._llm is not None:
            with contextlib.suppress(Exception):
                self._llm.set_cache(None)

    def _mark_chat_success(self) -> None:
        self.last_prompt_path = "chat_template"
        self.last_structured_output_fallback = False
        self.last_structured_output_fallback_reason = None

    def _mark_prompt_fallback(self, *, response_format: ResponseFormat | None, reason: str) -> None:
        self.last_prompt_path = "fallback_prompt"
        if _strict_response_format(response_format):
            self.last_structured_output_fallback = True
            self.last_structured_output_fallback_reason = reason

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

    def _message_dicts(
        self, messages: list[ChatMessage], *, disable_thinking: bool = False
    ) -> list[dict[str, str]]:
        result = [{"role": message.role, "content": message.content} for message in messages]
        if disable_thinking and self._model is not None and self._model.chat_format == "qwen":
            for item in reversed(result):
                if item["role"] == "user":
                    if "/no_think" not in item["content"]:
                        item["content"] = f"{item['content']}\n/no_think"
                    break
        return result

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
        if raw == "length":
            return "length"
        if raw == "error":
            return "error"
        if raw == "cancelled":
            return "cancelled"
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


def _build_prefix_adapter(
    base_cache: Any,
    llm: Any,
    policy: PrefixStatePolicy,
    on_error: Callable[[], None],
) -> Any:
    class PrefixCacheAdapter(base_cache):
        def __init__(self) -> None:
            super().__init__(capacity_bytes=policy.capacity_bytes)
            self.last_lookup: dict[str, object] = {}

        @property
        def cache_size(self) -> int:
            return policy.bytes

        def _find_longest_prefix_key(self, key: tuple[int, ...]) -> tuple[int, ...] | None:
            entry = policy.lookup(key)
            return entry.key if entry is not None else None

        def __getitem__(self, key: object) -> Any:
            try:
                raw_key: Any = key
                tokens = tuple(int(value) for value in raw_key)
                live_raw = getattr(llm, "input_ids", ())
                live_count = int(getattr(llm, "n_tokens", 0))
                live = tuple(int(value) for value in live_raw[:live_count])
                entry = policy.lookup(tokens)
                cached = entry.key if entry is not None else ()
                live_prefix = _common_prefix_length(live, tokens)
                cached_prefix = _common_prefix_length(tokens, cached)
                hit = cached_prefix > live_prefix
                self.last_lookup = {
                    "prompt_tokens": len(tokens),
                    "live_prefix_tokens": live_prefix,
                    "cached_prefix_tokens": cached_prefix,
                    "hit": hit,
                }
                if not hit or entry is None:
                    raise KeyError("prefix cache miss")
                return entry.state
            except KeyError:
                raise
            except Exception:
                on_error()
                raise KeyError("prefix cache miss") from None

        def __contains__(self, key: object) -> bool:
            try:
                self.__getitem__(key)
            except Exception:
                return False
            return True

        def __setitem__(self, key: object, state: Any) -> None:
            try:
                raw_key: Any = key
                tokens = tuple(int(value) for value in raw_key)
                nbytes = len(getattr(state, "llama_state", b""))
                for name in ("scores", "input_ids"):
                    value = getattr(state, name, None)
                    nbytes += int(getattr(value, "nbytes", 0))
                policy.insert(tokens, state, nbytes)
            except Exception:
                on_error()

    return PrefixCacheAdapter()


def _common_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    import numpy as np

    size = min(len(left), len(right))
    if size == 0:
        return ()
    lhs = np.asarray(left[:size], dtype=np.int64)
    rhs = np.asarray(right[:size], dtype=np.int64)
    mismatch = np.flatnonzero(lhs != rhs)
    end = int(mismatch[0]) if mismatch.size else size
    return left[:end]


def _common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return len(_common_prefix(left, right))


def _prefix_cache_enabled() -> bool:
    import os

    return os.environ.get("APRIL_RUNTIME_PREFIX_CACHE", "on").casefold() != "off"


def _strict_response_format(response_format: ResponseFormat | None) -> bool:
    return response_format is not None and response_format.type == "json_object"


def _is_structured_fallback_exception(exc: Exception) -> bool:
    message = str(exc).casefold()
    fallback_markers = (
        "response_format",
        "chat unsupported",
        "chat completion unsupported",
        "create_chat_completion",
        "not supported",
        "unsupported",
        "unexpected keyword",
        "unknown argument",
        "invalid keyword",
    )
    return any(marker in message for marker in fallback_markers)


def _fallback_reason(exc: Exception, *, response_format: ResponseFormat | None) -> str:
    if not _strict_response_format(response_format):
        return "chat_completion_error"
    if _is_structured_fallback_exception(exc):
        return "structured_output_unsupported"
    return "structured_output_error"


def _flatten_embedding(raw: Any) -> list[float]:
    # llama-cpp-python may return a flat vector or a list of per-token vectors.
    values = list(raw)
    if values and isinstance(values[0], (list, tuple)):
        return [float(component) for component in values[0]]
    return [float(component) for component in values]


def _flatten_embedding_batch(raw: Any) -> list[list[float]]:
    value = raw
    if isinstance(value, dict):
        value = value.get("data", value.get("embeddings"))
    rows = list(value)
    if rows and isinstance(rows[0], dict):
        rows = [row.get("embedding") for row in rows]
    if not all(isinstance(row, (list, tuple)) for row in rows):
        raise ValueError("native embedding batch result is malformed")
    return [[float(component) for component in row] for row in rows]
