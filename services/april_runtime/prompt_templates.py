from __future__ import annotations

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from april_common.errors import ConfigError
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.schemas import ChatMessage

GENERIC_TEMPLATE = """{% for message in messages -%}
{{ message.role | upper }}: {{ message.content }}
{% endfor -%}
ASSISTANT:"""

TEMPLATES_BY_FAMILY: dict[str, str] = {
    "granite": """{% for message in messages -%}
<|{{ message.role }}|>
{{ message.content }}
{% endfor -%}
<|assistant|>
""",
    "qwen": """{% for message in messages -%}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor -%}
<|im_start|>assistant
""",
}

NATIVE_TEMPLATE_METADATA_KEYS = ("tokenizer.chat_template", "chat_template")
CHAT_FORMAT_METADATA_KEYS = ("tokenizer.chat_format", "chat_format")


def select_template(model: ModelDefinition, metadata: dict[str, object] | None = None) -> str:
    if model.chat_format:
        return _template_for_format(model.chat_format, model=model)
    native_template = _metadata_native_template(metadata)
    if native_template is not None:
        return native_template
    metadata_format = _metadata_chat_format(metadata)
    if metadata_format is not None:
        return _template_for_format(metadata_format, model=model)
    inferred = _infer_chat_format(model.name)
    if inferred is not None:
        return _template_for_format(inferred, model=model)
    raise ConfigError(
        "Unsupported chat template for model.",
        {
            "model_id": model.id,
            "model_name": model.name,
            "hint": "Set chat_format to one of: generic, granite, qwen.",
        },
    )


def render_prompt(
    model: ModelDefinition,
    messages: list[ChatMessage],
    *,
    metadata: dict[str, object] | None = None,
) -> str:
    env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
    template = env.from_string(select_template(model, metadata))
    return template.render(messages=messages, add_generation_prompt=True)


def _template_for_format(chat_format: str, *, model: ModelDefinition) -> str:
    normalized = chat_format.lower()
    if normalized == "generic":
        return GENERIC_TEMPLATE
    if normalized in TEMPLATES_BY_FAMILY:
        return TEMPLATES_BY_FAMILY[normalized]
    raise ConfigError(
        "Unsupported chat template for model.",
        {
            "model_id": model.id,
            "model_name": model.name,
            "chat_format": chat_format,
            "hint": "Set chat_format to one of: generic, granite, qwen.",
        },
    )


def _metadata_native_template(metadata: dict[str, object] | None) -> str | None:
    if not metadata:
        return None
    for key in NATIVE_TEMPLATE_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _metadata_chat_format(metadata: dict[str, object] | None) -> str | None:
    if not metadata:
        return None
    for key in CHAT_FORMAT_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _infer_chat_format(name: str) -> str | None:
    normalized = name.casefold()
    if "granite" in normalized:
        return "granite"
    if "qwen" in normalized:
        return "qwen"
    return None
