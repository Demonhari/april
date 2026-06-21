from __future__ import annotations

from pathlib import Path

import pytest

from april_common.errors import ConfigError
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.prompt_templates import render_prompt
from services.april_runtime.schemas import ChatMessage


def model_data(path: str = "models/missing.gguf") -> dict[str, object]:
    return {
        "id": "april-brain",
        "name": "granite",
        "path": path,
        "backend": "llama_cpp",
        "role": "brain",
        "threads": 2,
        "context_size": 1024,
        "temperature": 0.2,
        "max_output_tokens": 64,
        "keep_loaded": False,
        "n_gpu_layers": 0,
        "n_batch": 128,
        "n_ubatch": 64,
        "use_mmap": True,
        "use_mlock": False,
        "chat_format": "generic",
        "idle_unload_seconds": 60,
        "priority": 10,
    }


def test_valid_registry(tmp_path: Path) -> None:
    registry = ModelRegistry.from_dict({"models": {"brain": model_data()}}, root=tmp_path)
    assert registry.get("april-brain").role == "brain"
    assert registry.get("april-brain").n_batch == 128
    assert registry.get("april-brain").priority == 10


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    data = {"models": {"one": model_data(), "two": {**model_data(), "name": "other"}}}
    with pytest.raises(ConfigError):
        ModelRegistry.from_dict(data, root=tmp_path)


def test_invalid_values_rejected(tmp_path: Path) -> None:
    data = {"models": {"brain": {**model_data(), "temperature": 3.0}}}
    with pytest.raises(ConfigError):
        ModelRegistry.from_dict(data, root=tmp_path)


def test_missing_model_path_state_is_unavailable(tmp_path: Path) -> None:
    registry = ModelRegistry.from_dict({"models": {"brain": model_data()}}, root=tmp_path)
    lifecycle = ModelLifecycle(registry)
    assert lifecycle.list_models()[0].state == "unavailable"
    assert lifecycle.list_models()[0].missing_path is True


def test_prompt_template_supports_generic_and_qwen_explicit_formats(tmp_path: Path) -> None:
    generic = ModelRegistry.from_dict({"models": {"brain": model_data()}}, root=tmp_path).get(
        "april-brain"
    )
    messages = [ChatMessage(role="user", content="hello")]
    assert "USER: hello" in render_prompt(generic, messages)

    qwen = generic.model_copy(update={"chat_format": "qwen"})
    assert "<|im_start|>user" in render_prompt(qwen, messages)


def test_granite_rendering_from_explicit_config(tmp_path: Path) -> None:
    model = ModelRegistry.from_dict(
        {"models": {"brain": {**model_data(), "chat_format": "granite"}}},
        root=tmp_path,
    ).get("april-brain")
    prompt = render_prompt(
        model,
        [
            ChatMessage(role="system", content="local only"),
            ChatMessage(role="user", content="hello"),
        ],
    )
    assert "<|system|>\nlocal only" in prompt
    assert "<|user|>\nhello" in prompt
    assert prompt.endswith("<|assistant|>")


def test_qwen_rendering_preserves_roles_and_content(tmp_path: Path) -> None:
    model = ModelRegistry.from_dict(
        {"models": {"brain": {**model_data(), "chat_format": "qwen"}}},
        root=tmp_path,
    ).get("april-brain")
    prompt = render_prompt(
        model,
        [
            ChatMessage(role="system", content="Use <literal> text"),
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="prior answer"),
        ],
    )
    assert "<|im_start|>system\nUse <literal> text<|im_end|>" in prompt
    assert "<|im_start|>user\nhello<|im_end|>" in prompt
    assert "<|im_start|>assistant\nprior answer<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant")


def test_explicit_chat_format_takes_precedence_over_metadata(tmp_path: Path) -> None:
    model = ModelRegistry.from_dict(
        {"models": {"brain": {**model_data(), "chat_format": "qwen"}}},
        root=tmp_path,
    ).get("april-brain")
    prompt = render_prompt(
        model,
        [ChatMessage(role="user", content="hello")],
        metadata={"chat_format": "granite"},
    )
    assert "<|im_start|>user" in prompt
    assert "<|user|>" not in prompt


def test_metadata_native_template_fallback(tmp_path: Path) -> None:
    model = ModelRegistry.from_dict(
        {"models": {"brain": {**model_data(), "chat_format": None, "name": "custom-local"}}},
        root=tmp_path,
    ).get("april-brain")
    prompt = render_prompt(
        model,
        [ChatMessage(role="user", content="hello")],
        metadata={
            "tokenizer.chat_template": (
                "{% for message in messages %}[{{ message.role }}]{{ message.content }}"
                "{% endfor %}[assistant]"
            )
        },
    )
    assert prompt == "[user]hello[assistant]"


def test_recognized_name_fallbacks(tmp_path: Path) -> None:
    granite = ModelRegistry.from_dict(
        {"models": {"brain": {**model_data(), "chat_format": None, "name": "Granite 3.3"}}},
        root=tmp_path,
    ).get("april-brain")
    qwen = granite.model_copy(update={"name": "Qwen3 coder"})
    assert "<|user|>" in render_prompt(granite, [ChatMessage(role="user", content="hello")])
    assert "<|im_start|>user" in render_prompt(qwen, [ChatMessage(role="user", content="hello")])


def test_unknown_model_without_template_fails_clearly(tmp_path: Path) -> None:
    model = ModelRegistry.from_dict(
        {"models": {"brain": {**model_data(), "chat_format": None, "name": "unknown-local"}}},
        root=tmp_path,
    ).get("april-brain")
    with pytest.raises(ConfigError, match="Unsupported chat template"):
        render_prompt(model, [ChatMessage(role="user", content="hello")])
