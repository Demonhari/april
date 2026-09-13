from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

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
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ModelState,
    Usage,
)

BackendFactory = Callable[[ModelDefinition], RuntimeBackend]
RuntimeStreamEventName = Literal["meta", "token", "usage", "done", "error"]


@dataclass(frozen=True, slots=True)
class RuntimeModelIdentity:
    """Immutable identity for one loaded model/adapter instance."""

    instance_id: str
    model_id: str
    candidate_id: str | None
    base_model_sha256: str
    adapter_id: str | None
    adapter_sha256: str | None
    configuration_sha256: str

    @property
    def is_candidate(self) -> bool:
        return self.candidate_id is not None


class ResourceLoadGate(Protocol):
    """Structural view of ResourceGovernor used to gate new specialist loads.

    ``assess_resident()`` considers RAM headroom and CPU load only; power and
    idle state deliberately never block an interactive model load.
    """

    def assess_resident(self) -> Any: ...  # GovernorDecision-shaped

    async def assess_resident_async(self) -> Any: ...  # GovernorDecision-shaped

    def assess_model_load(
        self,
        *,
        projected_resident_gb: float | None = None,
        current_resident_gb: float = 0.0,
        loaded_specialist_count: int = 0,
        max_loaded_specialist_count: int | None = None,
        speculative: bool = False,
    ) -> Any: ...  # GovernorDecision-shaped

    async def assess_model_load_async(
        self,
        *,
        projected_resident_gb: float | None = None,
        current_resident_gb: float = 0.0,
        loaded_specialist_count: int = 0,
        max_loaded_specialist_count: int | None = None,
        speculative: bool = False,
    ) -> Any: ...


@dataclass(slots=True)
class ModelRuntimeState:
    model: ModelDefinition
    identity: RuntimeModelIdentity | None = None
    base_model_id: str | None = None
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
    load_duration_ms: float | None = None
    last_used_monotonic: float = 0.0
    loaded_threads: int | None = None
    loaded_threads_batch: int | None = None


class ModelLifecycle:
    def __init__(
        self,
        registry: ModelRegistry,
        *,
        backend_factory: BackendFactory | None = None,
        root_backend: str | None = None,
        max_loaded_specialist_models: int = 2,
        governor: ResourceLoadGate | None = None,
        prefix_cache_enabled: bool = True,
    ) -> None:
        self.registry = registry
        self.root_backend = root_backend
        self.context_manager = ContextManager()
        self.max_loaded_specialist_models = max(0, max_loaded_specialist_models)
        # Optional resource gate for *new specialist loads* only. keep_loaded
        # models (the brain) and embedding models always load, and power/idle
        # never block an interactive load — the gate looks at RAM/CPU headroom.
        self.governor = governor
        self.prefix_cache_enabled = prefix_cache_enabled
        self._policy_lock = asyncio.Lock()
        self._states = {
            model.id: ModelRuntimeState(
                model=model,
                base_model_id=model.id,
                state=self._initial_state(model),
            )
            for model in registry.list()
        }
        self._candidate_instance_ids: set[str] = set()
        self._backend_factory = backend_factory or self._default_backend_factory

    def _initial_state(self, model: ModelDefinition) -> ModelState:
        if self.root_backend == "fake":
            return "unloaded"
        return "unloaded" if model.resolved_path(self.registry.root).exists() else "unavailable"

    def _default_backend_factory(self, model: ModelDefinition) -> RuntimeBackend:
        if self.root_backend == "fake" or model.backend == "fake":
            return FakeBackend()
        return LlamaCppBackend(prefix_cache_enabled=self.prefix_cache_enabled)

    def _is_specialist(self, model: ModelDefinition) -> bool:
        # keep_loaded models (e.g. the brain) and embedding-role models are
        # exempt from specialist load/eviction accounting.
        return not model.keep_loaded and model.role != "embedding"

    def embedding_model_id(self) -> str | None:
        for state in self._states.values():
            if state.model.role == "embedding":
                return state.model.id
        return None

    def configured_model_definitions(self) -> list[ModelDefinition]:
        return [state.model for state in self._states.values()]

    def get_state(self, model_id: str) -> ModelRuntimeState:
        try:
            return self._states[model_id]
        except KeyError as exc:
            raise NotFoundError("Model", {"model_id": model_id}) from exc

    def get_instance(self, instance_id: str) -> ModelRuntimeState:
        """Resolve either a registry model ID or an isolated instance ID."""

        return self.get_state(instance_id)

    def candidate_capability(self) -> dict[str, object]:
        return {
            "supported": bool(getattr(self._backend_factory, "supports_isolated_instances", True)),
            "backend_instance_isolation": "per_candidate_backend_object",
            "candidate_count": len(self._candidate_instance_ids),
        }

    def candidate_readiness(self) -> dict[str, object]:
        candidates: list[dict[str, object]] = []
        for instance_id in sorted(self._candidate_instance_ids):
            state = self._states[instance_id]
            identity = state.identity
            candidates.append(
                {
                    "instance_id": instance_id,
                    "model_id": state.base_model_id,
                    "candidate_id": identity.candidate_id if identity else None,
                    "state": state.state,
                    "integrity_state": ("verified" if identity is not None else "unknown"),
                    "base_model_sha256": identity.base_model_sha256 if identity else None,
                    "adapter_sha256": identity.adapter_sha256 if identity else None,
                    "configuration_sha256": (identity.configuration_sha256 if identity else None),
                    "load_error": state.load_error,
                    "active_requests": state.active_requests,
                }
            )
        baseline = next(
            (
                state
                for state in self._states.values()
                if not (state.identity and state.identity.is_candidate)
                and state.model.role == "brain"
            ),
            None,
        )
        return {
            **self.candidate_capability(),
            "baseline_model_instance": baseline.model.id if baseline else None,
            "candidate_instances": candidates,
            "candidate_integrity_state": (
                "mismatch"
                if any(item["integrity_state"] == "mismatch" for item in candidates)
                else ("verified" if candidates else "unknown")
            ),
        }

    async def prepare_candidate(
        self,
        *,
        model_id: str,
        candidate_id: str,
        adapter_path: Path,
        adapter_sha256: str,
        configuration_sha256: str,
        instance_id: str | None = None,
        load: bool = True,
    ) -> ModelRuntimeState:
        """Create and optionally load an immutable, separately addressable LoRA instance."""

        base = self.registry.get(model_id)
        if base.role == "embedding":
            raise ModelUnavailableError(model_id, "LoRA candidates cannot be embedding models.")
        resolved_base = base.resolved_path(self.registry.root)
        resolved_adapter = adapter_path.expanduser().resolve(strict=False)
        if not _is_within(resolved_adapter, self.registry.root):
            raise ModelUnavailableError(
                model_id,
                "Candidate adapter path is outside the Runtime model root.",
            )
        if not resolved_base.is_file() or resolved_base.is_symlink():
            raise ModelUnavailableError(model_id, "Base model file is unavailable.")
        if not resolved_adapter.is_file() or resolved_adapter.is_symlink():
            raise ModelUnavailableError(model_id, "Candidate adapter file is unavailable.")
        base_sha = await asyncio.to_thread(_sha256_cached, resolved_base)
        actual_adapter_sha = await asyncio.to_thread(_sha256_path, resolved_adapter)
        if actual_adapter_sha != adapter_sha256:
            raise ModelUnavailableError(model_id, "Candidate adapter hash mismatch.")
        if len(configuration_sha256) != 64:
            raise ModelUnavailableError(model_id, "Candidate configuration hash is invalid.")
        stable_id = instance_id or _candidate_instance_id(
            model_id,
            candidate_id,
            base_sha,
            actual_adapter_sha,
            configuration_sha256,
        )
        candidate_identity = RuntimeModelIdentity(
            instance_id=stable_id,
            model_id=model_id,
            candidate_id=candidate_id,
            base_model_sha256=base_sha,
            adapter_id=candidate_id,
            adapter_sha256=actual_adapter_sha,
            configuration_sha256=configuration_sha256,
        )
        existing = self._states.get(stable_id)
        if existing is not None:
            if existing.identity != candidate_identity:
                raise ModelUnavailableError(model_id, "Candidate instance identity is immutable.")
            if load:
                await self.load_candidate(stable_id)
            return existing
        candidate_model = base.model_copy(
            update={
                "id": stable_id,
                "path": resolved_base,
                "adapter_path": resolved_adapter,
            }
        )
        state = ModelRuntimeState(
            model=candidate_model,
            identity=candidate_identity,
            base_model_id=model_id,
            state=self._initial_state(candidate_model),
        )
        self._states[stable_id] = state
        self._candidate_instance_ids.add(stable_id)
        try:
            if load:
                await self.load_candidate(stable_id)
        except (asyncio.CancelledError, TimeoutError):
            raise
        except BaseException:
            # Keep a diagnostic state for readiness, but never expose a partially
            # loaded backend or let the failed candidate affect the baseline.
            await self._close_failed_state(state)
            raise
        return state

    async def unload_candidate(self, instance_id: str) -> ModelRuntimeState:
        state = self.get_state(instance_id)
        if state.identity is None or not state.identity.is_candidate:
            raise ModelUnavailableError(instance_id, "Model instance is not a candidate.")
        return await self.unload_model(instance_id)

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
            load_duration_ms=state.load_duration_ms,
            idle_unload_seconds=state.model.idle_unload_seconds,
            priority=state.model.priority,
            threads=state.loaded_threads or state.model.threads,
            threads_batch=(
                state.loaded_threads_batch or state.model.threads_batch or state.model.threads
            ),
            flash_attn=state.model.flash_attn,
            prefix_cache_mb=state.model.prefix_cache_mb,
            prefix_cache_min_tokens=state.model.prefix_cache_min_tokens,
            prefix_cache=(
                state.backend.prefix_cache_aggregate_diagnostics() if state.backend else None
            ),
            n_batch=state.model.n_batch,
            n_ubatch=state.model.n_ubatch,
            n_gpu_layers=state.model.n_gpu_layers,
            use_mmap=state.model.use_mmap,
            use_mlock=state.model.use_mlock,
            model_instance_id=state.identity.instance_id if state.identity else state.model.id,
            base_model_id=state.base_model_id or state.model.id,
            candidate_id=state.identity.candidate_id if state.identity else None,
            base_model_sha256=state.identity.base_model_sha256 if state.identity else None,
            adapter_sha256=state.identity.adapter_sha256 if state.identity else None,
            configuration_sha256=(state.identity.configuration_sha256 if state.identity else None),
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

    async def _check_resource_gate_async(self, state: ModelRuntimeState) -> None:
        """Refuse a *new* specialist load under RAM/CPU pressure.

        Already-loaded models are untouched; the caller sees an explicit
        ModelUnavailableError with the governor's reasons instead of a silent
        OOM later. Gate failures (probe errors) never block a load.
        """
        if self.governor is None or state.state in {"loaded", "loading"}:
            return
        try:
            assess_model_load = getattr(self.governor, "assess_model_load", None)
            assess_model_load_async = getattr(self.governor, "assess_model_load_async", None)
            kwargs = {
                "projected_resident_gb": self._projected_model_load_gb(state),
                "current_resident_gb": self._current_projected_resident_gb(),
                "loaded_specialist_count": self._loaded_specialist_count(),
                "max_loaded_specialist_count": self.max_loaded_specialist_models,
            }
            if callable(assess_model_load_async):
                decision = await assess_model_load_async(**kwargs)
            elif callable(assess_model_load):
                decision = await asyncio.to_thread(assess_model_load, **kwargs)
            else:
                assess_resident_async = getattr(self.governor, "assess_resident_async", None)
                if callable(assess_resident_async):
                    decision = await assess_resident_async()
                else:
                    decision = await asyncio.to_thread(self.governor.assess_resident)
        except Exception:
            return
        if getattr(decision, "allowed", True):
            return
        reasons = tuple(getattr(decision, "reasons", ()) or ())
        raise ModelUnavailableError(
            state.model.id,
            "Deferred specialist model load under resource pressure.",
            {"governor_reasons": list(reasons)},
        )

    def _projected_model_load_gb(self, state: ModelRuntimeState) -> float | None:
        if self.root_backend == "fake":
            return 0.0
        return state.model.projected_resident_gb(self.registry.root)

    def _current_projected_resident_gb(self) -> float:
        return sum(
            self._projected_model_load_gb(state) or 0.0
            for state in self._states.values()
            if state.state == "loaded"
        )

    def _loaded_specialist_count(self) -> int:
        return sum(
            1
            for state in self._states.values()
            if state.state == "loaded" and self._is_specialist(state.model)
        )

    async def load_model(
        self, model_id: str, *, generation_threads: int | None = None
    ) -> ModelRuntimeState:
        state = self.get_state(model_id)
        if generation_threads is not None and generation_threads < 1:
            raise ValueError("generation_threads must be positive")
        if self._is_specialist(state.model) and state.identity is None:
            await self._check_resource_gate_async(state)
            await self._enforce_lifecycle(target_model_id=model_id)
        elif state.identity is not None and state.identity.is_candidate:
            await self._check_resource_gate_async(state)
            if (
                self.max_loaded_specialist_models > 0
                and state.state != "loaded"
                and self._loaded_specialist_count() >= self.max_loaded_specialist_models
            ):
                raise ModelUnavailableError(
                    model_id,
                    "Candidate load would exceed the specialist instance limit; "
                    "baseline was left untouched.",
                    {"max_loaded_specialist_models": self.max_loaded_specialist_models},
                )
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
            # Resolve both the base GGUF path and any LoRA adapter path against
            # the registry root before handing the model to the backend, so the
            # backend loads the exact same absolute paths readiness validates
            # (readiness uses ModelDefinition.resolved_adapter_path(registry.root)).
            # A relative adapter_path must not be resolved against the process cwd.
            update: dict[str, object] = {"path": state.model.resolved_path(self.registry.root)}
            effective_threads = min(state.model.threads, generation_threads or state.model.threads)
            configured_batch = state.model.threads_batch or state.model.threads
            effective_threads_batch = min(configured_batch, generation_threads or configured_batch)
            update["threads"] = effective_threads
            update["threads_batch"] = effective_threads_batch
            try:
                resolved_adapter = state.model.resolved_adapter_path(self.registry.root)
            except (OSError, ValueError) as exc:
                state.state = "error"
                state.load_error = str(exc)
                raise ModelUnavailableError(
                    model_id,
                    "Unable to resolve active LoRA adapter.",
                    {"cause": str(exc)},
                ) from exc
            if (
                resolved_adapter is not None
                and self.root_backend != "fake"
                and state.model.backend != "fake"
                and not resolved_adapter.exists()
            ):
                state.state = "error"
                state.load_error = f"LoRA adapter file is missing: {resolved_adapter}"
                raise ModelUnavailableError(
                    model_id,
                    "Configured LoRA adapter path is missing.",
                    {"adapter_path": str(resolved_adapter)},
                )
            if resolved_adapter is not None:
                update["adapter_path"] = resolved_adapter
            resolved_model = state.model.model_copy(update=update)
            backend = self._backend_factory(resolved_model)
            if not getattr(backend, "supports_isolated_instances", True):
                state.state = "error"
                state.load_error = "backend_does_not_support_isolated_instances"
                raise ModelUnavailableError(
                    model_id, "Backend cannot provide isolated model instances."
                )
            if any(
                other.backend is backend
                for other_id, other in self._states.items()
                if other_id != model_id and other.backend is not None
            ):
                state.state = "error"
                state.load_error = "backend_instance_reused"
                raise ModelUnavailableError(
                    model_id, "Backend instance is shared with another model."
                )
            started = time.monotonic()
            try:
                await backend.load(resolved_model)
            except asyncio.CancelledError:
                state.state = "error"
                state.load_error = "candidate_load_cancelled"
                state.backend = None
                with contextlib.suppress(Exception):
                    await backend.unload()
                raise
            except Exception as exc:
                state.state = "error"
                state.load_error = str(exc)
                state.backend = None
                with contextlib.suppress(Exception):
                    await backend.unload()
                raise ModelUnavailableError(
                    model_id, "Unable to load model.", {"cause": str(exc)}
                ) from exc
            state.backend = backend
            state.state = "loaded"
            if state.identity is None:
                state.identity = RuntimeModelIdentity(
                    instance_id=state.model.id,
                    model_id=state.model.id,
                    candidate_id=None,
                    base_model_sha256=(
                        await asyncio.to_thread(_sha256_cached, resolved_model.path)
                        if resolved_model.path.is_file()
                        else hashlib.sha256(b"").hexdigest()
                    ),
                    adapter_id=(
                        resolved_model.adapter_path.name if resolved_model.adapter_path else None
                    ),
                    adapter_sha256=(
                        await asyncio.to_thread(_sha256_path, resolved_model.adapter_path)
                        if resolved_model.adapter_path is not None
                        and resolved_model.adapter_path.is_file()
                        else None
                    ),
                    configuration_sha256=_configuration_hash(resolved_model),
                )
            state.loaded_threads = resolved_model.threads
            state.loaded_threads_batch = resolved_model.threads_batch or resolved_model.threads
            state.loaded_at = utc_now_iso()
            state.load_duration_ms = (time.monotonic() - started) * 1000
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
                state.loaded_threads = None
                state.loaded_threads_batch = None
                state.unloaded_at = utc_now_iso()
            return state

    async def cleanup(self) -> None:
        for model_id in list(self._states):
            await self.unload_model(model_id)

    async def _close_failed_state(self, state: ModelRuntimeState) -> None:
        if state.backend is not None:
            with contextlib.suppress(Exception):
                await state.backend.unload()
        state.backend = None
        state.state = "error"

    async def load_candidate(
        self, instance_id: str, *, timeout_seconds: float | None = None
    ) -> ModelRuntimeState:
        try:
            if timeout_seconds is None:
                return await self.load_model(instance_id)
            return await asyncio.wait_for(self.load_model(instance_id), timeout_seconds)
        except (asyncio.CancelledError, TimeoutError):
            with contextlib.suppress(Exception):
                await self.unload_candidate(instance_id)
            raise

    async def generate(self, request: ChatRequest) -> ChatResponse:
        request_id = request.request_id or str(uuid.uuid4())
        request_started = time.monotonic()
        state = await self.load_model(
            request.model_id, generation_threads=request.generation_threads
        )
        loaded_at = time.monotonic()
        if state.backend is None:
            raise ModelUnavailableError(request.model_id, "Model backend is not available.")
        # Reserve the loaded instance before the first await after load_model.
        # A thread-budget change may reload an idle model, so context preparation
        # must count as active use just like backend generation does.
        state.active_requests += 1
        try:
            options = effective_generation_options(state.model, request.options)
            metadata = state.backend.prompt_metadata()
            context, reserved_output_tokens = await self.context_manager.fit_with_reservation(
                model=state.model,
                backend=state.backend,
                messages=request.messages,
                max_output_tokens=options.max_output_tokens,
                metadata=metadata,
            )
        except BaseException:
            state.active_requests = max(0, state.active_requests - 1)
            raise
        disable_thinking = options.enable_thinking is False
        prompt = render_prompt(
            state.model,
            context.messages,
            metadata=metadata,
            disable_thinking=disable_thinking,
        )
        context_ready_at = time.monotonic()
        lock = (
            state.generation_lock
            if not state.backend.supports_concurrent_generation
            else _NoopLock()
        )
        async with lock:
            lock_acquired_at = time.monotonic()
            thread_budget_applied = await self._apply_thread_budget(
                state, request.generation_threads
            )
            try:
                if request.options.enable_thinking is not None:
                    result = await _generate_messages_compat(
                        state.backend,
                        prompt,
                        messages=context.messages,
                        temperature=options.temperature,
                        max_output_tokens=reserved_output_tokens,
                        top_p=options.top_p,
                        stop=options.stop,
                        seed=options.seed,
                        response_format=request.response_format,
                        disable_thinking=disable_thinking,
                        prompt_tokens=context.input_tokens,
                    )
                else:
                    result = await _generate_messages_compat(
                        state.backend,
                        prompt,
                        messages=context.messages,
                        temperature=options.temperature,
                        max_output_tokens=reserved_output_tokens,
                        top_p=options.top_p,
                        stop=options.stop,
                        seed=options.seed,
                        response_format=request.response_format,
                        prompt_tokens=context.input_tokens,
                    )
            except Exception as exc:
                state.state = "error"
                state.load_error = str(exc)
                state.generation_errors += 1
                # A failed generation may have left the backend in an unknown
                # native state. Close it before exposing the error so a later
                # load cannot retain a live, poisoned instance.
                await self._close_failed_state(state)
                raise ModelUnavailableError(
                    request.model_id, "Generation failed.", {"cause": str(exc)}
                ) from exc
            finally:
                state.active_requests = max(0, state.active_requests - 1)
            generation_done_at = time.monotonic()
            elapsed = max(generation_done_at - lock_acquired_at, 0.000_001)
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
        timing = {
            **state.backend.timing_diagnostics(),
            "prompt_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "threads": state.loaded_threads,
            "threads_batch": state.loaded_threads_batch,
            "thread_budget_applied": thread_budget_applied,
            "load_ms": max((loaded_at - request_started) * 1000, 0.0),
            "context_ms": max((context_ready_at - loaded_at) * 1000, 0.0),
            "lock_wait_ms": max((lock_acquired_at - context_ready_at) * 1000, 0.0),
            "generation_ms": max((generation_done_at - lock_acquired_at) * 1000, 0.0),
            "total_ms": max((generation_done_at - request_started) * 1000, 0.0),
        }
        _add_derived_timing(timing)
        usage = usage.model_copy(update={"timing": timing})
        warnings = ["Context was truncated."] if context.truncated else []
        reservation_reduced = reserved_output_tokens < options.max_output_tokens
        if reservation_reduced:
            warnings.append(
                f"Output reservation reduced to {reserved_output_tokens} tokens to fit the context."
            )
        if "context_truncated_without_persisted_summary" in context.context_warning_codes:
            warnings.append(
                "Older context was omitted and no persisted Core conversation summary was supplied."
            )
        structured_fallback = bool(getattr(state.backend, "last_structured_output_fallback", False))
        if structured_fallback:
            warnings.append(
                "Structured output used prompt fallback; native chat/structured support "
                "was not verified for this request."
            )
        diagnostics = {
            "prompt_path": getattr(state.backend, "last_prompt_path", None),
            "structured_output_fallback": structured_fallback,
            "structured_output_fallback_reason": getattr(
                state.backend,
                "last_structured_output_fallback_reason",
                None,
            ),
            "context_size_used": state.model.context_size,
            "context_budget": context.metadata(),
            "reserved_output_tokens": reserved_output_tokens,
            "output_reservation_reduced": reservation_reduced,
            "timing": timing,
            "prefix_cache": state.backend.prefix_cache_diagnostics(),
        }
        return ChatResponse(
            request_id=request_id,
            model_id=request.model_id,
            content=result.text,
            finish_reason=result.finish_reason,
            usage=usage,
            context_truncated=context.truncated,
            warnings=warnings,
            diagnostics={key: value for key, value in diagnostics.items() if value is not None},
        )

    async def count_message_tokens(self, model_id: str, messages: list[ChatMessage]) -> int:
        """Count a complete chat payload with the loaded model tokenizer."""

        state = await self.load_model(model_id)
        if state.backend is None:
            raise ModelUnavailableError(model_id, "Model backend is not available.")
        prompt = render_prompt(
            state.model,
            messages,
            metadata=state.backend.prompt_metadata(),
        )
        return await state.backend.count_tokens(prompt)

    async def embed(self, text: str, *, model_id: str | None = None) -> tuple[str, list[float]]:
        resolved_id, vectors = await self.embed_many([text], model_id=model_id)
        return resolved_id, vectors[0]

    async def embed_many(
        self,
        texts: list[str],
        *,
        model_id: str | None = None,
    ) -> tuple[str, list[list[float]]]:
        resolved_id = model_id or self.embedding_model_id()
        if resolved_id is None:
            raise ModelUnavailableError(
                "embedding",
                "No embedding-role model is registered.",
            )
        state = self.get_state(resolved_id)
        if state.model.role != "embedding":
            raise ModelUnavailableError(
                resolved_id,
                "Model is not an embedding-role model.",
                {"role": state.model.role},
            )
        loaded = await self.load_model(resolved_id)
        if loaded.backend is None:
            raise ModelUnavailableError(resolved_id, "Model backend is not available.")
        loaded.active_requests += 1
        try:
            vectors = await loaded.backend.embed_many(texts)
            if len(vectors) != len(texts):
                raise ValueError("embedding backend returned the wrong vector count")
            dimensions = len(vectors[0]) if vectors else 0
            if dimensions < 1:
                raise ValueError("embedding backend returned a missing vector")
            if any(len(vector) != dimensions for vector in vectors):
                raise ValueError("embedding backend returned inconsistent dimensions")
            if any(not math.isfinite(value) for vector in vectors for value in vector):
                raise ValueError("embedding backend returned non-finite values")
        except AprilError:
            raise
        except Exception as exc:
            loaded.generation_errors += 1
            raise ModelUnavailableError(
                resolved_id,
                "Embedding batch failed validation.",
                {"cause": str(exc)},
            ) from exc
        finally:
            loaded.active_requests = max(0, loaded.active_requests - 1)
        loaded.last_used_at = utc_now_iso()
        loaded.last_used_monotonic = time.monotonic()
        return resolved_id, vectors

    async def stream(
        self, request: ChatRequest
    ) -> AsyncIterator[tuple[RuntimeStreamEventName, dict[str, object]]]:
        request_started = time.monotonic()
        state = await self.load_model(
            request.model_id, generation_threads=request.generation_threads
        )
        loaded_at = time.monotonic()
        if state.backend is None:
            raise ModelUnavailableError(request.model_id, "Model backend is not available.")
        state.active_requests += 1
        try:
            options = effective_generation_options(state.model, request.options)
            metadata = state.backend.prompt_metadata()
            context, reserved_output_tokens = await self.context_manager.fit_with_reservation(
                model=state.model,
                backend=state.backend,
                messages=request.messages,
                max_output_tokens=options.max_output_tokens,
                metadata=metadata,
            )
        except BaseException:
            state.active_requests = max(0, state.active_requests - 1)
            raise
        disable_thinking = options.enable_thinking is False
        prompt = render_prompt(
            state.model,
            context.messages,
            metadata=metadata,
            disable_thinking=disable_thinking,
        )
        context_ready_at = time.monotonic()
        input_tokens = context.input_tokens
        output_tokens = 0
        lock = (
            state.generation_lock
            if not state.backend.supports_concurrent_generation
            else _NoopLock()
        )
        async with lock:
            lock_acquired_at = time.monotonic()
            thread_budget_applied = await self._apply_thread_budget(
                state, request.generation_threads
            )
            reservation_reduced = reserved_output_tokens < options.max_output_tokens
            reservation_warnings = (
                [
                    "Output reservation reduced to "
                    f"{reserved_output_tokens} tokens to fit the context."
                ]
                if reservation_reduced
                else []
            )
            yield (
                "meta",
                {
                    "context_truncated": context.truncated,
                    "context_budget": context.metadata(),
                    "reserved_output_tokens": reserved_output_tokens,
                    "output_reservation_reduced": reservation_reduced,
                    "warnings": reservation_warnings,
                },
            )
            try:
                if request.options.enable_thinking is not None:
                    token_stream = _stream_messages_compat(
                        state.backend,
                        prompt,
                        messages=context.messages,
                        temperature=options.temperature,
                        max_output_tokens=reserved_output_tokens,
                        top_p=options.top_p,
                        stop=options.stop,
                        seed=options.seed,
                        response_format=request.response_format,
                        disable_thinking=disable_thinking,
                        prompt_tokens=context.input_tokens,
                    )
                else:
                    token_stream = _stream_messages_compat(
                        state.backend,
                        prompt,
                        messages=context.messages,
                        temperature=options.temperature,
                        max_output_tokens=reserved_output_tokens,
                        top_p=options.top_p,
                        stop=options.stop,
                        seed=options.seed,
                        response_format=request.response_format,
                        prompt_tokens=context.input_tokens,
                    )
                async for token in token_stream:
                    if output_tokens == 0:
                        first_token_at = time.monotonic()
                    output_tokens += len(await state.backend.tokenize(token))
                    yield "token", {"text": token}
            except asyncio.CancelledError:
                yield "done", {"finish_reason": "cancelled"}
                raise
            except Exception as exc:
                state.state = "error"
                state.load_error = str(exc)
                state.generation_errors += 1
                await self._close_failed_state(state)
                yield "error", {"code": "GENERATION_FAILED", "message": "Generation failed."}
                return
            finally:
                # Reset per-request lifecycle state on every exit path (normal
                # completion, client disconnect, cancellation, or error) so a
                # cancelled stream cannot strand active_requests or make the model
                # look idle the instant it was last used.
                state.active_requests = max(0, state.active_requests - 1)
                state.last_used_at = utc_now_iso()
                state.last_used_monotonic = time.monotonic()
            generation_done_at = time.monotonic()
        state.generations += 1
        state.input_tokens += input_tokens
        state.output_tokens += output_tokens
        elapsed = max(generation_done_at - lock_acquired_at, 0.000_001)
        state.recent_latency_ms = elapsed * 1000
        state.recent_tokens_per_second = output_tokens / elapsed
        state.backend.finish_timing_diagnostics(input_tokens, output_tokens)
        timing = {
            **state.backend.timing_diagnostics(),
            "prompt_tokens": input_tokens,
            "output_tokens": output_tokens,
            "threads": state.loaded_threads,
            "threads_batch": state.loaded_threads_batch,
            "thread_budget_applied": thread_budget_applied,
            "load_ms": max((loaded_at - request_started) * 1000, 0.0),
            "context_ms": max((context_ready_at - loaded_at) * 1000, 0.0),
            "lock_wait_ms": max((lock_acquired_at - context_ready_at) * 1000, 0.0),
            "generation_ms": max((generation_done_at - lock_acquired_at) * 1000, 0.0),
            "total_ms": max((generation_done_at - request_started) * 1000, 0.0),
        }
        if "first_token_at" in locals():
            timing["ttft_ms"] = max((first_token_at - request_started) * 1000, 0.0)
        _add_derived_timing(timing)
        yield (
            "usage",
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "prompt_path": getattr(state.backend, "last_prompt_path", None),
                "structured_output_fallback": bool(
                    getattr(state.backend, "last_structured_output_fallback", False)
                ),
                "structured_output_fallback_reason": getattr(
                    state.backend,
                    "last_structured_output_fallback_reason",
                    None,
                ),
                "context_size_used": state.model.context_size,
                "context_budget": context.metadata(),
                "reserved_output_tokens": reserved_output_tokens,
                "output_reservation_reduced": reserved_output_tokens < options.max_output_tokens,
                "warnings": (
                    [
                        "Output reservation reduced to "
                        f"{reserved_output_tokens} tokens to fit the context."
                    ]
                    if reserved_output_tokens < options.max_output_tokens
                    else []
                ),
                "timing": timing,
                "prefix_cache": state.backend.prefix_cache_diagnostics(),
            },
        )
        yield "done", {"finish_reason": "stop"}

    async def _enforce_lifecycle(self, *, target_model_id: str) -> None:
        async with self._policy_lock:
            await self._unload_idle_specialists()
            await self._evict_for_capacity(target_model_id=target_model_id)

    async def _apply_thread_budget(
        self, state: ModelRuntimeState, generation_threads: int | None
    ) -> bool:
        if state.backend is None:
            return False
        configured = state.model.threads
        configured_batch = state.model.threads_batch or configured
        desired = min(configured, generation_threads or configured)
        desired_batch = min(configured_batch, generation_threads or configured_batch)
        if state.loaded_threads == desired and state.loaded_threads_batch == desired_batch:
            return True
        applied = state.backend.apply_thread_budget(desired, desired_batch)
        if applied:
            state.loaded_threads = desired
            state.loaded_threads_batch = desired_batch
        return applied

    async def _unload_idle_specialists(self) -> None:
        now = time.monotonic()
        for state in self._states.values():
            if not self._is_specialist(state.model) or state.state != "loaded":
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
        if not self._is_specialist(target.model):
            return
        loaded_specialists = [
            state
            for state in self._states.values()
            if state.state == "loaded" and self._is_specialist(state.model)
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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_BASE_DIGEST_CACHE: dict[tuple[str, int, int, int], str] = {}


def _sha256_cached(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino)
    cached = _BASE_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    digest = _sha256_path(path)
    _BASE_DIGEST_CACHE[key] = digest
    return digest


def _candidate_instance_id(
    model_id: str,
    candidate_id: str,
    base_sha256: str,
    adapter_sha256: str,
    configuration_sha256: str,
) -> str:
    material = "|".join((model_id, candidate_id, base_sha256, adapter_sha256, configuration_sha256))
    suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"candidate:{model_id}:{candidate_id}:{suffix}"


def _configuration_hash(model: ModelDefinition) -> str:
    payload = model.model_dump(mode="json", exclude={"path", "adapter_path", "id"})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _add_derived_timing(timing: dict[str, object]) -> None:
    prompt_ms = timing.get("prompt_eval_ms")
    eval_ms = timing.get("eval_ms")
    prompt_eval_tokens = timing.get("prompt_eval_tokens")
    eval_tokens = timing.get("eval_tokens")
    prompt_tokens = timing.get("prompt_tokens")
    if (
        isinstance(prompt_ms, (int, float))
        and prompt_ms > 0
        and isinstance(prompt_eval_tokens, (int, float))
    ):
        timing["prompt_eval_tokens_per_second"] = (
            float(prompt_eval_tokens) / float(prompt_ms) * 1000.0
        )
    if isinstance(eval_ms, (int, float)) and eval_ms > 0 and isinstance(eval_tokens, (int, float)):
        timing["eval_tokens_per_second"] = float(eval_tokens) / float(eval_ms) * 1000.0
    if (
        isinstance(prompt_tokens, (int, float))
        and prompt_tokens > 0
        and isinstance(prompt_eval_tokens, (int, float))
    ):
        timing["prompt_reuse_ratio"] = min(
            1.0, max(0.0, 1.0 - float(prompt_eval_tokens) / float(prompt_tokens))
        )


async def _generate_messages_compat(backend: RuntimeBackend, prompt: str, **kwargs: Any) -> Any:
    method = backend.generate_messages
    parameters: Mapping[str, inspect.Parameter]
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "prompt_tokens" not in parameters:
        kwargs.pop("prompt_tokens", None)
    return await method(prompt, **kwargs)


def _stream_messages_compat(
    backend: RuntimeBackend, prompt: str, **kwargs: Any
) -> AsyncIterator[str]:
    method = backend.stream_messages
    parameters: Mapping[str, inspect.Parameter]
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "prompt_tokens" not in parameters:
        kwargs.pop("prompt_tokens", None)
    return method(prompt, **kwargs)
