from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Literal

from april_common.errors import AprilError, ModelUnavailableError, NotFoundError
from april_common.time import utc_now_iso
from services.april_runtime.backend import RuntimeBackend
from services.april_runtime.context_manager import ContextManager
from services.april_runtime.fake_backend import FakeBackend
from services.april_runtime.generation import effective_generation_options
from services.april_runtime.llama_cpp_backend import LlamaCppBackend
from services.april_runtime.model_registry import ModelDefinition, ModelRegistry
from services.april_runtime.prompt_templates import render_prompt
from services.april_runtime.schemas import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ModelState,
    Usage,
)

BackendFactory = Callable[[ModelDefinition], RuntimeBackend]
RuntimeStreamEventName = Literal["meta", "token", "usage", "done", "error"]


@dataclass(slots=True)
class ModelRuntimeState:
    model: ModelDefinition
    state: ModelState = "unloaded"
    backend: RuntimeBackend | None = None
    lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used_at: str | None = None
    load_error: str | None = None
    generations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    active_requests: int = 0
    generation_errors: int = 0
    recent_latency_ms: float | None = None
    recent_tokens_per_second: float | None = None
    loaded_at: str | None = None
    unloaded_at: str | None = None
    last_used_monotonic: float = 0.0


class ModelLifecycle:
    def __init__(
        self,
        registry: ModelRegistry,
        *,
        backend_factory: BackendFactory | None = None,
        root_backend: str | None = None,
        max_loaded_specialist_models: int = 2,
    ) -> None:
        self.registry = registry
        self.root_backend = root_backend
        self.context_manager = ContextManager()
        self.max_loaded_specialist_models = max(0, max_loaded_specialist_models)
        self._policy_lock = asyncio.Lock()
        self._states = {
            model.id: ModelRuntimeState(model=model, state=self._initial_state(model))
            for model in registry.list()
        }
        self._backend_factory = backend_factory or self._default_backend_factory

    def _initial_state(self, model: ModelDefinition) -> ModelState:
        if self.root_backend == "fake":
            return "unloaded"
        return "unloaded" if model.resolved_path(self.registry.root).exists() else "unavailable"

    def _default_backend_factory(self, model: ModelDefinition) -> RuntimeBackend:
        if self.root_backend == "fake" or model.backend == "fake":
            return FakeBackend()
        return LlamaCppBackend()

    def get_state(self, model_id: str) -> ModelRuntimeState:
        try:
            return self._states[model_id]
        except KeyError as exc:
            raise NotFoundError("Model", {"model_id": model_id}) from exc

    def list_models(self) -> list[ModelInfo]:
        return [self._model_info(state) for state in self._states.values()]

    def _model_info(self, state: ModelRuntimeState) -> ModelInfo:
        path = state.model.resolved_path(self.registry.root)
        return ModelInfo(
            id=state.model.id,
            name=state.model.name,
            role=state.model.role,
            backend=self.root_backend or state.model.backend,
            path=str(path),
            state=state.state,
            keep_loaded=state.model.keep_loaded,
            context_size=state.model.context_size,
            temperature=state.model.temperature,
            max_output_tokens=state.model.max_output_tokens,
            last_used_at=state.last_used_at,
            load_error=state.load_error,
            generations=state.generations,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            missing_path=not path.exists(),
            active_requests=state.active_requests,
            generation_errors=state.generation_errors,
            recent_latency_ms=state.recent_latency_ms,
            recent_tokens_per_second=state.recent_tokens_per_second,
            loaded_at=state.loaded_at,
            unloaded_at=state.unloaded_at,
            idle_unload_seconds=state.model.idle_unload_seconds,
            priority=state.model.priority,
        )

    def policy_snapshot(self) -> dict[str, object]:
        return {
            "max_loaded_specialist_models": self.max_loaded_specialist_models,
            "idle_unload_enabled": any(
                state.model.idle_unload_seconds is not None for state in self._states.values()
            ),
            "eviction": "priority_then_lru",
        }

    async def preload(self) -> None:
        for state in self._states.values():
            if state.model.keep_loaded:
                try:
                    await self.load_model(state.model.id)
                except AprilError:
                    continue

    async def load_model(self, model_id: str) -> ModelRuntimeState:
        state = self.get_state(model_id)
        if not state.model.keep_loaded:
            await self._enforce_lifecycle(target_model_id=model_id)
        async with state.lifecycle_lock:
            if state.state == "loaded":
                return state
            if state.state == "loading":
                return state
            if state.state == "unavailable" and self.root_backend != "fake":
                raise ModelUnavailableError(
                    model_id,
                    "Configured model path is missing.",
                    {"path": str(state.model.resolved_path(self.registry.root))},
                )
            state.state = "loading"
            state.load_error = None
            resolved_model = state.model.model_copy(
                update={"path": state.model.resolved_path(self.registry.root)}
            )
            backend = self._backend_factory(resolved_model)
            try:
                await backend.load(resolved_model)
            except Exception as exc:
                state.state = "error"
                state.load_error = str(exc)
                state.backend = None
                raise ModelUnavailableError(
                    model_id, "Unable to load model.", {"cause": str(exc)}
                ) from exc
            state.backend = backend
            state.state = "loaded"
            state.loaded_at = utc_now_iso()
            state.unloaded_at = None
            state.last_used_monotonic = time.monotonic()
            return state

    async def unload_model(self, model_id: str) -> ModelRuntimeState:
        state = self.get_state(model_id)
        async with state.lifecycle_lock:
            if state.active_requests > 0:
                raise ModelUnavailableError(
                    model_id,
                    "Cannot unload model while active requests are running.",
                    {"active_requests": state.active_requests},
                )
            if state.state in {"unloaded", "unavailable"}:
                return state
            if state.backend is None:
                state.state = "unloaded"
                return state
            state.state = "unloading"
            try:
                await state.backend.unload()
            finally:
                state.backend = None
                state.state = self._initial_state(state.model)
                state.unloaded_at = utc_now_iso()
            return state

    async def cleanup(self) -> None:
        for model_id in list(self._states):
            await self.unload_model(model_id)

    async def generate(self, request: ChatRequest) -> ChatResponse:
        request_id = request.request_id or str(uuid.uuid4())
        state = await self.load_model(request.model_id)
        if state.backend is None:
            raise ModelUnavailableError(request.model_id, "Model backend is not available.")
        options = effective_generation_options(state.model, request.options)
        context = await self.context_manager.fit(
            model=state.model,
            backend=state.backend,
            messages=request.messages,
            max_output_tokens=options.max_output_tokens,
        )
        prompt = render_prompt(state.model, context.messages)
        lock = (
            state.generation_lock
            if not state.backend.supports_concurrent_generation
            else _NoopLock()
        )
        async with lock:
            start = time.monotonic()
            try:
                state.active_requests += 1
                result = await state.backend.generate(
                    prompt,
                    temperature=options.temperature,
                    max_output_tokens=options.max_output_tokens,
                    top_p=options.top_p,
                    stop=options.stop,
                    seed=options.seed,
                )
            except Exception as exc:
                state.state = "error"
                state.load_error = str(exc)
                state.generation_errors += 1
                raise ModelUnavailableError(
                    request.model_id, "Generation failed.", {"cause": str(exc)}
                ) from exc
            finally:
                state.active_requests = max(0, state.active_requests - 1)
            elapsed = max(time.monotonic() - start, 0.000_001)
        state.last_used_at = utc_now_iso()
        state.last_used_monotonic = time.monotonic()
        state.generations += 1
        state.input_tokens += result.input_tokens
        state.output_tokens += result.output_tokens
        state.recent_latency_ms = elapsed * 1000
        state.recent_tokens_per_second = result.output_tokens / elapsed
        usage = Usage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        )
        warnings = ["Context was truncated."] if context.truncated else []
        return ChatResponse(
            request_id=request_id,
            model_id=request.model_id,
            content=result.text,
            finish_reason=result.finish_reason,
            usage=usage,
            context_truncated=context.truncated,
            warnings=warnings,
        )

    async def stream(
        self, request: ChatRequest
    ) -> AsyncIterator[tuple[RuntimeStreamEventName, dict[str, object]]]:
        state = await self.load_model(request.model_id)
        if state.backend is None:
            raise ModelUnavailableError(request.model_id, "Model backend is not available.")
        options = effective_generation_options(state.model, request.options)
        context = await self.context_manager.fit(
            model=state.model,
            backend=state.backend,
            messages=request.messages,
            max_output_tokens=options.max_output_tokens,
        )
        prompt = render_prompt(state.model, context.messages)
        input_tokens = await state.backend.count_tokens(prompt)
        output_tokens = 0
        start = time.monotonic()
        lock = (
            state.generation_lock
            if not state.backend.supports_concurrent_generation
            else _NoopLock()
        )
        async with lock:
            state.active_requests += 1
            yield "meta", {"context_truncated": context.truncated}
            try:
                async for token in state.backend.stream(
                    prompt,
                    temperature=options.temperature,
                    max_output_tokens=options.max_output_tokens,
                    top_p=options.top_p,
                    stop=options.stop,
                    seed=options.seed,
                ):
                    output_tokens += len(await state.backend.tokenize(token))
                    yield "token", {"text": token}
            except asyncio.CancelledError:
                yield "done", {"finish_reason": "cancelled"}
                raise
            except Exception as exc:
                state.state = "error"
                state.load_error = str(exc)
                state.generation_errors += 1
                yield "error", {"code": "GENERATION_FAILED", "message": "Generation failed."}
                return
            finally:
                state.active_requests = max(0, state.active_requests - 1)
        state.last_used_at = utc_now_iso()
        state.last_used_monotonic = time.monotonic()
        state.generations += 1
        state.input_tokens += input_tokens
        state.output_tokens += output_tokens
        elapsed = max(time.monotonic() - start, 0.000_001)
        state.recent_latency_ms = elapsed * 1000
        state.recent_tokens_per_second = output_tokens / elapsed
        yield (
            "usage",
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )
        yield "done", {"finish_reason": "stop"}

    async def _enforce_lifecycle(self, *, target_model_id: str) -> None:
        async with self._policy_lock:
            await self._unload_idle_specialists()
            await self._evict_for_capacity(target_model_id=target_model_id)

    async def _unload_idle_specialists(self) -> None:
        now = time.monotonic()
        for state in self._states.values():
            if state.model.keep_loaded or state.state != "loaded":
                continue
            if state.active_requests > 0:
                continue
            idle_seconds = state.model.idle_unload_seconds
            if idle_seconds is None:
                continue
            if state.last_used_monotonic and now - state.last_used_monotonic >= idle_seconds:
                await self.unload_model(state.model.id)

    async def _evict_for_capacity(self, *, target_model_id: str) -> None:
        if self.max_loaded_specialist_models <= 0:
            return
        target = self.get_state(target_model_id)
        if target.model.keep_loaded:
            return
        loaded_specialists = [
            state
            for state in self._states.values()
            if state.state == "loaded" and not state.model.keep_loaded
        ]
        if target.state != "loaded":
            projected_count = len(loaded_specialists) + 1
        else:
            projected_count = len(loaded_specialists)
        while projected_count > self.max_loaded_specialist_models:
            candidates = [
                state
                for state in loaded_specialists
                if state.model.id != target_model_id and state.active_requests == 0
            ]
            if not candidates:
                raise ModelUnavailableError(
                    target_model_id,
                    "No inactive specialist model is available for eviction.",
                    {"max_loaded_specialist_models": self.max_loaded_specialist_models},
                )
            victim = sorted(
                candidates,
                key=lambda item: (
                    item.model.priority,
                    item.last_used_monotonic,
                    item.model.id,
                ),
            )[0]
            await self.unload_model(victim.model.id)
            loaded_specialists.remove(victim)
            projected_count -= 1


class _NoopLock:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None
