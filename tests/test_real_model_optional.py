from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import ChatMessage, ChatRequest


@pytest.mark.asyncio
async def test_optional_real_gguf_load_generate_stream_unload(tmp_path: Path) -> None:
    model_path = os.environ.get("APRIL_TEST_GGUF_PATH")
    if not model_path:
        pytest.skip("APRIL_TEST_GGUF_PATH is not set.")
    gguf = Path(model_path).expanduser().resolve()
    if not gguf.exists():
        pytest.skip(f"APRIL_TEST_GGUF_PATH does not exist: {gguf}")
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "real": {
                        "id": "april-real-test",
                        "name": "real-test",
                        "path": str(gguf),
                        "backend": "llama_cpp",
                        "role": "brain",
                        "threads": 2,
                        "context_size": 1024,
                        "temperature": 0.0,
                        "max_output_tokens": 16,
                        "keep_loaded": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry.from_file(config_path, root=tmp_path)
    lifecycle = ModelLifecycle(registry, root_backend="llama_cpp")
    await lifecycle.load_model("april-real-test")
    response = await lifecycle.generate(
        ChatRequest(
            model_id="april-real-test",
            messages=[ChatMessage(role="user", content="Reply with one short sentence.")],
            max_tokens=8,
        )
    )
    assert response.content
    events = [
        event
        async for event in lifecycle.stream(
            ChatRequest(
                model_id="april-real-test",
                messages=[ChatMessage(role="user", content="Say hello.")],
                max_tokens=8,
            )
        )
    ]
    assert any(name == "token" for name, _payload in events)
    await lifecycle.unload_model("april-real-test")
