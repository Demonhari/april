from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from services.april_runtime.llama_cpp_backend import LlamaCppBackend
from services.april_runtime.model_registry import ModelDefinition

CAPTURED_TEST_GGUF_PATH = os.environ.get("APRIL_TEST_GGUF_PATH")
_REAL_OPTIONAL_AVAILABLE = (
    bool(CAPTURED_TEST_GGUF_PATH) and importlib.util.find_spec("llama_cpp") is not None
)


@pytest.mark.skipif(
    not _REAL_OPTIONAL_AVAILABLE,
    reason="set APRIL_TEST_GGUF_PATH and install llama-cpp-python",
)
@pytest.mark.asyncio
async def test_real_prefix_cache_and_thread_budget() -> None:
    assert CAPTURED_TEST_GGUF_PATH is not None
    model_path = Path(CAPTURED_TEST_GGUF_PATH).expanduser().resolve()
    if not model_path.is_file():
        pytest.skip("APRIL_TEST_GGUF_PATH is not a file")
    model = ModelDefinition(
        id="optional-real",
        name="optional-real",
        path=model_path,
        backend="llama_cpp",
        role="brain",
        threads=4,
        threads_batch=4,
        context_size=4096,
        temperature=0.0,
        max_output_tokens=16,
        prefix_cache_mb=64,
        prefix_cache_min_tokens=16,
        chat_format="generic",
    )
    backend = LlamaCppBackend()
    await backend.load(model)
    try:
        prefix = "system guidance " * 700
        first = await backend.generate(
            f"{prefix}\nA", temperature=0.0, max_output_tokens=4, seed=17
        )
        await backend.generate(
            "different family " * 700, temperature=0.0, max_output_tokens=4, seed=17
        )
        restored = await backend.generate(
            f"{prefix}\nA prime", temperature=0.0, max_output_tokens=4, seed=17
        )
        diagnostics = backend.prefix_cache_diagnostics()
        timing = backend.timing_diagnostics()
        assert diagnostics.get("restored") is True
        assert timing.get("prompt_eval_tokens", 0) < 0.25 * timing.get("prompt_tokens", 1)
        assert backend.apply_thread_budget(2, 3)
        module = backend._llama_module
        assert module.llama_n_threads(backend._llm.ctx) == 2
        assert module.llama_n_threads_batch(backend._llm.ctx) == 3
        short = await backend.generate("short", temperature=0.0, max_output_tokens=1, seed=17)
        del short, first
        short_diagnostics = backend.prefix_cache_diagnostics()
        assert short_diagnostics.get("attached") is False
        assert "lookup_performed" not in short_diagnostics
        cache_enabled = backend.prefix_cache_diagnostics().get("enabled") is True
        original_attach = backend._attach_prefix_cache
        backend._attach_prefix_cache = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected before lookup")
        )
        with pytest.raises(RuntimeError):
            await backend.generate("injected", temperature=0.0, max_output_tokens=1, seed=17)
        backend._attach_prefix_cache = original_attach
        assert cache_enabled
        assert backend.prefix_cache_diagnostics().get("enabled") is True
        cache_off = LlamaCppBackend(prefix_cache_enabled=False)
        await cache_off.load(model.model_copy(update={"prefix_cache_mb": 0}))
        try:
            fresh = await cache_off.generate(
                f"{prefix}\nA prime", temperature=0.0, max_output_tokens=4, seed=17
            )
            print("optional greedy cache equality:", restored.text == fresh.text)
        finally:
            await cache_off.unload()
    finally:
        await backend.unload()


@pytest.mark.skipif(
    not _REAL_OPTIONAL_AVAILABLE,
    reason="set APRIL_TEST_GGUF_PATH and install llama-cpp-python",
)
@pytest.mark.asyncio
async def test_real_prefix_cache_short_first_long_next() -> None:
    assert CAPTURED_TEST_GGUF_PATH is not None
    model_path = Path(CAPTURED_TEST_GGUF_PATH).expanduser().resolve()
    if not model_path.is_file():
        pytest.skip("APRIL_TEST_GGUF_PATH is not a file")
    model = ModelDefinition(
        id="optional-real-guard",
        name="optional-real-guard",
        path=model_path,
        backend="llama_cpp",
        role="brain",
        threads=4,
        threads_batch=4,
        context_size=4096,
        temperature=0.0,
        max_output_tokens=2048,
        prefix_cache_mb=1024,
        prefix_cache_min_tokens=16,
        chat_format="generic",
    )
    backend = LlamaCppBackend()
    await backend.load(model)
    try:
        await backend.generate(
            "short prefix token " * 150,
            temperature=0.0,
            max_output_tokens=1,
        )
        await backend.generate(
            "long prefix token " * 1000,
            temperature=0.0,
            max_output_tokens=2048,
        )
        diagnostics = backend.prefix_cache_diagnostics()
        assert diagnostics.get("attached") is True
        assert diagnostics.get("attach_skipped_reason") != "projected_oversize"
    finally:
        await backend.unload()
