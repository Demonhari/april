"""Optional adapter for an explicitly configured local Colibri server.

Colibri is deliberately treated as a model runtime only.  The adapter sends
messages and generation controls, never APRIL's privileged tool registry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.backend import BackendHealth, GenerationResult, RuntimeBackend
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.schemas import ChatMessage, ResponseFormat

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_UNSET = object()


def validate_colibri_url(url: str | None) -> str:
    if not url:
        raise ValueError("Colibri endpoint is not configured")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("Colibri endpoint must use an HTTP loopback host")
    if parsed.username or parsed.password:
        raise ValueError("Colibri endpoint must not contain credentials")
    return url.rstrip("/")


def colibri_manifest_digest(model: ModelDefinition, model_dir: Path) -> str:
    """Hash the configured metadata manifest, never the model weight directory."""
    entries: list[dict[str, object]] = []
    for relative in sorted(model.colibri_expected_files):
        candidate = (model_dir / relative).resolve(strict=False)
        try:
            candidate.relative_to(model_dir.resolve(strict=False))
        except ValueError:
            continue
        try:
            stat = candidate.stat()
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                remaining = 1024 * 1024
                while remaining > 0:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    digest.update(chunk)
                    remaining -= len(chunk)
        except OSError:
            entries.append({"path": relative, "missing": True})
            continue
        entries.append({"path": relative, "size": stat.st_size, "digest": digest.hexdigest()})
    material = json.dumps(
        {
            "model_id": model.id,
            "model_name": model.colibri_model_name or model.name,
            "expected_files": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


Tokenizer = Callable[[str], Sequence[int]]


class ColibriBackend(RuntimeBackend):
    """OpenAI-compatible, local-only Colibri backend.

    ``transport`` and ``tokenizer`` are dependency-injection seams for offline
    tests.  In production the transport is ordinary httpx to the configured
    loopback endpoint and the tokenizer is loaded from the configured local
    model directory when the optional ``tokenizers`` package is available.
    """

    supports_concurrent_generation = False
    supports_native_batch_embeddings = False

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        tokenizer: Tokenizer | None = None,
        tokenizer_path: Path | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = validate_colibri_url(base_url) if base_url else None
        self.model_name = model_name
        self._tokenizer = tokenizer
        self._tokenizer_path = tokenizer_path
        self._timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._loaded_model: ModelDefinition | None = None
        self.last_structured_output_fallback = False

    async def load(self, model: ModelDefinition) -> None:
        if model.backend != "colibri":
            raise RuntimeUnavailableError("Colibri backend received a non-Colibri model.")
        if model.artifact_kind != "colibri_model_directory":
            raise RuntimeUnavailableError("Colibri requires a model directory artifact.")
        model_dir = model.path
        if not model_dir.is_absolute():
            model_dir = model_dir.resolve(strict=False)
        if not model_dir.is_dir():
            raise RuntimeUnavailableError("Configured Colibri model directory is unavailable.")
        for relative in model.colibri_expected_files:
            candidate = (model_dir / relative).resolve(strict=False)
            if not _is_within(candidate, model_dir) or not candidate.is_file():
                raise RuntimeUnavailableError("Configured Colibri model metadata is incomplete.")
        if self.base_url is None:
            try:
                self.base_url = validate_colibri_url(model.colibri_base_url)
            except ValueError as exc:
                raise RuntimeUnavailableError("Colibri endpoint is not configured.") from exc
        if self.model_name is None:
            self.model_name = model.colibri_model_name or model.name
        if self._tokenizer is None:
            configured_tokenizer = model.colibri_tokenizer_path
            if configured_tokenizer is not None:
                configured_tokenizer = Path(configured_tokenizer)
            if configured_tokenizer is not None and not configured_tokenizer.is_absolute():
                configured_tokenizer = model_dir / configured_tokenizer
            self._tokenizer_path = configured_tokenizer or self._tokenizer_path
            self._load_optional_tokenizer()
        health = await self.health()
        if not health.ok:
            raise RuntimeUnavailableError("Colibri is configured but unavailable.")
        await self._validate_model_name()
        if self._tokenizer is None:
            raise RuntimeUnavailableError(
                "Colibri exact local tokenizer is unavailable; configure tokenizer.json."
            )
        self._loaded_model = model

    async def unload(self) -> None:
        self._loaded_model = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

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
        return await self.generate_messages(
            prompt,
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
        )

    def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        return self.stream_messages(
            prompt,
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
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
        disable_thinking: bool | None = None,
        prompt_tokens: int | None = None,
    ) -> GenerationResult:
        del prompt
        payload = self._payload(
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
            response_format=response_format,
            disable_thinking=disable_thinking,
            stream=False,
        )
        response = await self._request("POST", "/v1/chat/completions", json=payload)
        data = _json_object(response)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = _content_text(message.get("content", ""))
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RuntimeUnavailableError("Colibri returned malformed generation JSON.") from exc
        usage = cast(
            Mapping[str, Any],
            data.get("usage") if isinstance(data.get("usage"), dict) else {},
        )
        input_tokens = _positive_int(usage.get("prompt_tokens"), prompt_tokens or 0)
        output_tokens = _positive_int(usage.get("completion_tokens"), len(content.split()))
        finish = choice.get("finish_reason")
        return GenerationResult(
            text=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="length" if finish == "length" else "stop",
        )

    def stream_messages(
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
        disable_thinking: bool | None = None,
        prompt_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        del prompt, prompt_tokens

        async def _stream() -> AsyncIterator[str]:
            payload = self._payload(
                messages=messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop=stop,
                seed=seed,
                response_format=response_format,
                disable_thinking=disable_thinking,
                stream=True,
            )
            client = await self._http_client()
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        raise RuntimeUnavailableError("Colibri streaming request failed.")
                    completed = False
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            completed = True
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" not in chunk and "usage" in chunk:
                                continue
                            delta = chunk["choices"][0].get("delta", {})
                            text = _content_text(delta.get("content", ""))
                        except (ValueError, KeyError, IndexError, TypeError) as exc:
                            raise RuntimeUnavailableError(
                                "Colibri returned malformed streaming JSON."
                            ) from exc
                        if text:
                            yield text
                    if not completed:
                        raise RuntimeUnavailableError("Colibri stream ended before [DONE].")
            except httpx.HTTPError as exc:
                raise RuntimeUnavailableError("Colibri is configured but unavailable.") from exc

        return _stream()

    async def tokenize(self, text: str) -> list[int]:
        if self._tokenizer is not None:
            return list(self._tokenizer(text))
        if self._tokenizer_path is not None:
            self._load_optional_tokenizer()
        if self._tokenizer is not None:
            return list(self._tokenizer(text))
        raise RuntimeUnavailableError(
            "Colibri exact local tokenizer is unavailable; /tokenize is not supported."
        )

    async def health(self) -> BackendHealth:
        if self.base_url is None:
            return BackendHealth(False, "configured but unavailable: endpoint is not configured")
        try:
            response = await self._request("GET", "/health")
            data = _json_object(response)
            status = data.get("status", data.get("ok"))
            if status not in {True, "ok", "healthy", "ready"}:
                return BackendHealth(False, "configured but unavailable: health is not ready")
            return BackendHealth(True, "Colibri loopback service is healthy")
        except Exception:
            return BackendHealth(False, "configured but unavailable")

    def capabilities(self) -> dict[str, object]:
        return {
            "backend": "colibri",
            "local_only": True,
            "native_tools_forwarded": False,
            "embeddings": False,
            "streaming": True,
            "tokenizer": self._tokenizer is not None,
            "per_request_seed": False,
            "thinking": True,
            "response_format": "json_object_and_schema",
        }

    def _payload(self, *, messages: list[ChatMessage], **values: object) -> dict[str, object]:
        seed = values.pop("seed", None)
        del seed
        response_format = values.pop("response_format", None)
        disable_thinking = values.pop("disable_thinking", _UNSET)
        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": [message.model_dump() for message in messages],
        }
        payload.update({key: value for key, value in values.items() if value is not None})
        if isinstance(response_format, ResponseFormat) and response_format.type == "json_object":
            payload["response_format"] = _colibri_response_format(response_format)
        if disable_thinking is not None and disable_thinking is not _UNSET:
            payload["enable_thinking"] = not bool(disable_thinking)
        # Deliberately no ``tools`` or ``tool_choice`` keys.  APRIL owns tools.
        return payload

    async def _validate_model_name(self) -> None:
        if not self.model_name:
            return
        try:
            response = await self._request("GET", "/v1/models")
            data = _json_object(response)
            models = data.get("data")
            if not isinstance(models, list):
                raise RuntimeUnavailableError("Colibri model identity response is malformed.")
            ids = {
                str(item.get("id"))
                for item in models
                if isinstance(item, dict) and item.get("id") is not None
            }
            if self.model_name not in ids:
                raise RuntimeUnavailableError(
                    "Colibri model identity does not match configuration."
                )
        except RuntimeUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeUnavailableError("Colibri model identity could not be verified.") from exc

    async def _http_client(self) -> httpx.AsyncClient:
        if self.base_url is None:
            raise RuntimeUnavailableError("Colibri endpoint is not configured.")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                transport=self._transport,
            )
        return self._client

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            client = await self._http_client()
            response = await client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError("Colibri is configured but unavailable.") from exc
        if response.status_code >= 400:
            raise RuntimeUnavailableError("Colibri returned an error.")
        return response

    def _load_optional_tokenizer(self) -> None:
        if self._tokenizer is not None or self._tokenizer_path is None:
            return
        if not self._tokenizer_path.is_file():
            return
        try:
            from tokenizers import Tokenizer

            local = Tokenizer.from_file(str(self._tokenizer_path))
            self._tokenizer = local.encode
        except (ImportError, OSError, ValueError):
            return


def _colibri_response_format(response_format: ResponseFormat) -> dict[str, object] | None:
    if response_format.type != "json_object":
        return None
    if response_format.json_schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"schema": response_format.json_schema},
    }


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeUnavailableError("Colibri returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise RuntimeUnavailableError("Colibri returned a malformed JSON object.")
    return data


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item.get("text", "")) for item in value if isinstance(item, dict))
    return ""


def _positive_int(value: object, fallback: int) -> int:
    return int(value) if isinstance(value, int) and value >= 0 else fallback


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True
