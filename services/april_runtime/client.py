from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.schemas import (
    CandidateRuntimeRequest,
    CandidateRuntimeResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbedBatchRequest,
    EmbedBatchResponse,
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
        generation_thread_provider: Callable[[], int] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token
        self.generation_thread_provider = generation_thread_provider

    def _generation_threads(self) -> int | None:
        if self.generation_thread_provider is None:
            return None
        try:
            value = int(self.generation_thread_provider())
        except Exception:
            return None
        return value if value > 0 else None

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
            generation_threads=self._generation_threads(),
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

    async def embed_many(
        self,
        texts: list[str],
        *,
        model_id: str | None = None,
    ) -> list[list[float]]:
        """Use the typed batch endpoint, with narrow legacy-endpoint fallback."""
        request_id = str(uuid.uuid4())
        request = EmbedBatchRequest(
            texts=texts,
            model_id=model_id,
            request_id=request_id,
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/runtime/embed/batch",
                    json=request.model_dump(),
                    headers=self.headers,
                )
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
        unsupported = response.status_code in {404, 405} or _unsupported_capability(response)
        if unsupported:
            return [await self.embed(text, model_id=model_id) for text in request.texts]
        if response.status_code >= 400:
            raise RuntimeUnavailableError(
                "April Runtime returned an error.",
                _response_payload(response),
            )
        try:
            payload = EmbedBatchResponse.model_validate(response.json())
        except Exception as exc:
            raise RuntimeUnavailableError(
                "April Runtime returned a malformed embedding batch response.",
                {"status_code": response.status_code},
            ) from exc
        if payload.request_id != request_id:
            raise RuntimeUnavailableError(
                "April Runtime returned a mismatched embedding batch request ID."
            )
        if payload.count != len(request.texts):
            raise RuntimeUnavailableError("April Runtime returned the wrong embedding count.")
        return payload.embeddings

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

    async def load(
        self,
        model_id: str,
        *,
        request_id: str | None = None,
        generation_threads: int | None = None,
    ) -> ModelOperationResponse:
        return await self._model_operation(
            "load",
            model_id,
            request_id=request_id,
            generation_threads=(
                generation_threads if generation_threads is not None else self._generation_threads()
            ),
        )

    async def unload(
        self, model_id: str, *, request_id: str | None = None
    ) -> ModelOperationResponse:
        return await self._model_operation(
            "unload", model_id, request_id=request_id, generation_threads=None
        )

    async def prepare_candidate(
        self,
        *,
        model_id: str,
        candidate_id: str,
        adapter_path: str,
        adapter_sha256: str,
        configuration_sha256: str,
        instance_id: str | None = None,
        load: bool = True,
        request_id: str | None = None,
    ) -> CandidateRuntimeResponse:
        request = CandidateRuntimeRequest(
            model_id=model_id,
            candidate_id=candidate_id,
            adapter_path=adapter_path,
            adapter_sha256=adapter_sha256,
            configuration_sha256=configuration_sha256,
            instance_id=instance_id,
            load=load,
            request_id=request_id,
        )
        return await self._candidate_operation("prepare", request)

    async def unload_candidate(
        self,
        *,
        instance_id: str,
        model_id: str = "candidate",
        candidate_id: str = "candidate",
        adapter_sha256: str = "0" * 64,
        configuration_sha256: str = "0" * 64,
        request_id: str | None = None,
    ) -> CandidateRuntimeResponse:
        request = CandidateRuntimeRequest(
            model_id=model_id,
            candidate_id=candidate_id,
            adapter_path="candidate",
            adapter_sha256=adapter_sha256,
            configuration_sha256=configuration_sha256,
            instance_id=instance_id,
            load=False,
            request_id=request_id,
        )
        return await self._candidate_operation("unload", request)

    async def _candidate_operation(
        self, operation: str, request: CandidateRuntimeRequest
    ) -> CandidateRuntimeResponse:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/runtime/candidates/{operation}",
                    json=request.model_dump(),
                    headers=self.headers,
                )
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
        if response.status_code >= 400:
            raise RuntimeUnavailableError(
                "April Runtime returned an error.", _response_payload(response)
            )
        try:
            return CandidateRuntimeResponse.model_validate(response.json())
        except Exception as exc:
            raise RuntimeUnavailableError(
                "April Runtime returned a malformed candidate response."
            ) from exc

    async def _model_operation(
        self,
        operation: str,
        model_id: str,
        *,
        request_id: str | None,
        generation_threads: int | None,
    ) -> ModelOperationResponse:
        request = LoadModelRequest(
            model_id=model_id,
            generation_threads=generation_threads,
            request_id=request_id,
        )
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
            generation_threads=self._generation_threads(),
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


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError:
        return {"status_code": response.status_code}
    return value if isinstance(value, dict) else {"status_code": response.status_code}


def _unsupported_capability(response: httpx.Response) -> bool:
    if response.status_code < 400:
        return False
    payload = _response_payload(response)
    error = payload.get("error")
    return isinstance(error, dict) and error.get("code") == "UNSUPPORTED_CAPABILITY"
