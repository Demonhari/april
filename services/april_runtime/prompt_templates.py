from __future__ import annotations

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

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


def select_template(model: ModelDefinition) -> str:
    if model.chat_format:
        configured = model.chat_format.lower()
        if configured in TEMPLATES_BY_FAMILY:
            return TEMPLATES_BY_FAMILY[configured]
    return GENERIC_TEMPLATE


def render_prompt(model: ModelDefinition, messages: list[ChatMessage]) -> str:
    env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
    template = env.from_string(select_template(model))
    return template.render(messages=messages)
