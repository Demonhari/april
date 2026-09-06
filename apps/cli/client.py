from __future__ import annotations

from typing import Any

import httpx


class ApiOfflineError(Exception):
    pass


class ApiResponseError(ApiOfflineError):
    """The API is reachable but returned an error status.

    Distinct from a transport failure so callers with a local fallback (e.g.
    ``april mute``) never treat a denied/invalid request as "API unavailable".
    """


class AprilApiClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    async def get(
        self, path: str, *, params: dict[str, Any] | None = None, auth: bool = True
    ) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers=self.headers if auth else None,
                )
        except httpx.HTTPError as exc:
            raise ApiOfflineError(self.startup_hint()) from exc
        return self._json(response)

    async def post(self, path: str, payload: dict[str, Any], *, auth: bool = True) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    headers=self.headers if auth else None,
                )
        except httpx.HTTPError as exc:
            raise ApiOfflineError(self.startup_hint()) from exc
        return self._json(response)

    async def delete(self, path: str) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(f"{self.base_url}{path}", headers=self.headers)
        except httpx.HTTPError as exc:
            raise ApiOfflineError(self.startup_hint()) from exc
        return self._json(response)

    async def briefing(self) -> Any:
        return await self.get("/scheduler/briefing/preview")

    def _json(self, response: httpx.Response) -> Any:
        data = response.json()
        if response.status_code >= 400:
            message = data.get("error", {}).get("message", "APRIL API returned an error.")
            if not isinstance(message, str):
                message = "APRIL API returned an error."
            raise ApiResponseError(message)
        return data

    def startup_hint(self) -> str:
        return (
            "APRIL API is offline. Use the controlled startup path: run `run april "
            "audit verify --json`, then `run april readiness`, and finally `run april "
            "start --preflight`. Direct `make run-runtime`/`make run-api` commands "
            "are development diagnostics and do not replace the audit gate."
        )
