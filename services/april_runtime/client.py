from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    GenerationOptions,
    LoadModelRequest,
    ModelOperationResponse,
    ResponseFormat,
)


class RuntimeClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token

    @property
    def headers(self) -> dict[str, str] | None:
        if not self.token:
            return None
        return {"Authorization": f"Bearer {self.token}"}

    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
        response_format: ResponseFormat | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        request = ChatRequest(
            model_id=model_id,
            messages=messages,
            options=options or GenerationOptions(),
            response_format=response_format,
            request_id=request_id,
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/runtime/chat",
                    json=request.model_dump(),
                    headers=self.headers,
                )
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
        if response.status_code >= 400:
            raise RuntimeUnavailableError("April Runtime returned an error.", response.json())
        return ChatResponse.model_validate(response.json())

    async def embed(self, text: str, *, model_id: str | None = None) -> list[float]:
        request = EmbedRequest(text=text, model_id=model_id)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/runtime/embed",
                    json=request.model_dump(),
                    headers=self.headers,
                )
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
        if response.status_code >= 400:
            raise RuntimeUnavailableError("April Runtime returned an error.", response.json())
        return EmbedResponse.model_validate(response.json()).embedding

    async def models(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/runtime/models", headers=self.headers)
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
        if response.status_code >= 400:
            raise RuntimeUnavailableError("April Runtime returned an error.", response.json())
        return response.json()

    async def health(self, *, timeout: float | None = None) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                response = await client.get(f"{self.base_url}/runtime/health", headers=self.headers)
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
        if response.status_code >= 400:
            raise RuntimeUnavailableError("April Runtime returned an error.", response.json())
        return response.json()

    async def load(self, model_id: str, *, request_id: str | None = None) -> ModelOperationResponse:
        return await self._model_operation("load", model_id, request_id=request_id)

    async def unload(
        self, model_id: str, *, request_id: str | None = None
    ) -> ModelOperationResponse:
        return await self._model_operation("unload", model_id, request_id=request_id)

    async def _model_operation(
        self,
        operation: str,
        model_id: str,
        *,
        request_id: str | None,
    ) -> ModelOperationResponse:
        request = LoadModelRequest(model_id=model_id, request_id=request_id)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/runtime/models/{operation}",
                    json=request.model_dump(),
                    headers=self.headers,
                )
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
        if response.status_code >= 400:
            raise RuntimeUnavailableError("April Runtime returned an error.", response.json())
        return ModelOperationResponse.model_validate(response.json())

    async def stream(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
        response_format: ResponseFormat | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        request = ChatRequest(
            model_id=model_id,
            messages=messages,
            options=options or GenerationOptions(),
            response_format=response_format,
            request_id=request_id,
        )
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout) as client,
                client.stream(
                    "POST",
                    f"{self.base_url}/runtime/stream",
                    json=request.model_dump(),
                    headers=self.headers,
                ) as response,
            ):
                if response.status_code >= 400:
                    raise RuntimeUnavailableError(
                        "April Runtime returned an error.",
                        {"status_code": response.status_code},
                    )
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
