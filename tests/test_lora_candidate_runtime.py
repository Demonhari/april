from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from april_common.errors import ModelUnavailableError
from services.april_runtime.backend import BackendHealth, GenerationResult, RuntimeBackend
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import ChatMessage, ChatRequest


class InstanceBackend(RuntimeBackend):
    def __init__(self, *, fail_candidate: bool = False, slow_load: bool = False) -> None:
        self.loaded_model = None
        self.fail_candidate = fail_candidate
        self.slow_load = slow_load
        self.unloaded = False

    async def load(self, model) -> None:  # type: ignore[no-untyped-def]
        if self.slow_load:
            await asyncio.sleep(1)
        if self.fail_candidate and model.id.startswith("candidate:"):
            raise RuntimeError("candidate load failed")
        self.loaded_model = model

    async def unload(self) -> None:
        self.unloaded = True
        self.loaded_model = None

    async def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        del prompt, kwargs
        await asyncio.sleep(0)
        return GenerationResult("instance", 1, 1)

    async def stream(self, prompt: str, **kwargs: object):  # type: ignore[no-untyped-def]
        del prompt, kwargs
        yield "instance"

    async def tokenize(self, text: str) -> list[int]:
        return text.split()

    async def health(self) -> BackendHealth:
        return BackendHealth(True, "ok")


def _registry(root: Path) -> ModelRegistry:
    return ModelRegistry.from_dict(
        {
            "models": {
                "brain": {
                    "id": "april-brain",
                    "name": "local",
                    "path": "models/base.gguf",
                    "backend": "fake",
                    "role": "brain",
                    "chat_format": "generic",
                    "threads": 1,
                    "context_size": 1024,
                    "temperature": 0.2,
                    "max_output_tokens": 32,
                    "keep_loaded": True,
                }
            }
        },
        root=root,
    )


def _files(root: Path) -> tuple[Path, Path]:
    base = root / "models" / "base.gguf"
    adapter = root / "data" / "evolution" / "adapters" / "candidate.gguf"
    base.parent.mkdir(parents=True)
    adapter.parent.mkdir(parents=True)
    base.write_bytes(b"base-model")
    adapter.write_bytes(b"candidate-adapter")
    return base, adapter


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.asyncio
async def test_baseline_and_candidate_are_concurrent_and_do_not_share_adapter(
    tmp_path: Path,
) -> None:
    base, adapter = _files(tmp_path)
    backends: list[InstanceBackend] = []

    def factory(_model):  # type: ignore[no-untyped-def]
        backend = InstanceBackend()
        backends.append(backend)
        return backend

    lifecycle = ModelLifecycle(_registry(tmp_path), backend_factory=factory, root_backend="fake")
    baseline = await lifecycle.load_model("april-brain")
    candidate = await lifecycle.prepare_candidate(
        model_id="april-brain",
        candidate_id="adapter-v1",
        adapter_path=adapter,
        adapter_sha256=_sha(adapter),
        configuration_sha256="1" * 64,
    )
    assert baseline.backend is not candidate.backend
    assert baseline.identity is not None
    assert baseline.identity.adapter_sha256 is None
    assert candidate.identity is not None
    assert candidate.identity.base_model_sha256 == _sha(base)
    assert candidate.identity.adapter_sha256 == _sha(adapter)
    assert candidate.model.adapter_path == adapter
    assert baseline.model.adapter_path is None
    request = ChatRequest(
        model_id="april-brain",
        messages=[ChatMessage(role="user", content="baseline")],
    )
    candidate_request = request.model_copy(update={"model_id": candidate.model.id})
    await asyncio.gather(lifecycle.generate(request), lifecycle.generate(candidate_request))
    assert len(backends) == 2
    assert backends[0].loaded_model is not None
    assert backends[1].loaded_model is not None
    assert backends[0].loaded_model.adapter_path is None
    assert backends[1].loaded_model.adapter_path == adapter


@pytest.mark.asyncio
async def test_candidate_identity_is_immutable_and_hash_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    _base, adapter = _files(tmp_path)
    lifecycle = ModelLifecycle(
        _registry(tmp_path), backend_factory=lambda _model: InstanceBackend(), root_backend="fake"
    )
    first = await lifecycle.prepare_candidate(
        model_id="april-brain",
        candidate_id="adapter-v1",
        adapter_path=adapter,
        adapter_sha256=_sha(adapter),
        configuration_sha256="2" * 64,
        instance_id="candidate-fixed",
    )
    same = await lifecycle.prepare_candidate(
        model_id="april-brain",
        candidate_id="adapter-v1",
        adapter_path=adapter,
        adapter_sha256=_sha(adapter),
        configuration_sha256="2" * 64,
        instance_id="candidate-fixed",
    )
    assert same is first
    with pytest.raises(ModelUnavailableError, match="identity is immutable"):
        await lifecycle.prepare_candidate(
            model_id="april-brain",
            candidate_id="adapter-v2",
            adapter_path=adapter,
            adapter_sha256=_sha(adapter),
            configuration_sha256="2" * 64,
            instance_id="candidate-fixed",
        )
    with pytest.raises(ModelUnavailableError, match="hash mismatch"):
        await lifecycle.prepare_candidate(
            model_id="april-brain",
            candidate_id="adapter-v3",
            adapter_path=adapter,
            adapter_sha256="3" * 64,
            configuration_sha256="3" * 64,
        )


@pytest.mark.asyncio
async def test_failed_candidate_load_leaves_loaded_baseline_untouched(tmp_path: Path) -> None:
    _base, adapter = _files(tmp_path)
    lifecycle = ModelLifecycle(
        _registry(tmp_path),
        backend_factory=lambda _model: InstanceBackend(fail_candidate=True),
        root_backend="fake",
    )
    baseline = await lifecycle.load_model("april-brain")
    with pytest.raises(ModelUnavailableError, match="Unable to load model"):
        await lifecycle.prepare_candidate(
            model_id="april-brain",
            candidate_id="adapter-failing",
            adapter_path=adapter,
            adapter_sha256=_sha(adapter),
            configuration_sha256="4" * 64,
        )
    assert lifecycle.get_state("april-brain").backend is baseline.backend
    assert lifecycle.get_state("april-brain").state == "loaded"


@pytest.mark.asyncio
async def test_candidate_timeout_unloads_without_stranding_active_state(tmp_path: Path) -> None:
    _base, adapter = _files(tmp_path)
    lifecycle = ModelLifecycle(
        _registry(tmp_path),
        backend_factory=lambda _model: InstanceBackend(slow_load=True),
        root_backend="fake",
    )
    state = await lifecycle.prepare_candidate(
        model_id="april-brain",
        candidate_id="adapter-timeout",
        adapter_path=adapter,
        adapter_sha256=_sha(adapter),
        configuration_sha256="5" * 64,
        load=False,
    )
    with pytest.raises(asyncio.TimeoutError):
        await lifecycle.load_candidate(state.model.id, timeout_seconds=0.01)
    assert lifecycle.get_state(state.model.id).state == "unloaded"
    assert lifecycle.get_state("april-brain").state == "unloaded"
