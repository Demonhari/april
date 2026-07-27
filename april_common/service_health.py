from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class ServiceHealthResult:
    ok: bool
    status_code: int | None
    reason: str
    message: str


def probe_service_health(
    url: str,
    *,
    bearer_token: str | None = None,
    timeout: float = 1.0,
) -> ServiceHealthResult:
    """Probe one HTTP endpoint without following redirects or exposing secrets."""
    headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else None
    try:
        response = httpx.get(
            url,
            headers=headers,
            timeout=max(0.01, timeout),
            follow_redirects=False,
        )
    except httpx.TimeoutException:
        return ServiceHealthResult(
            ok=False,
            status_code=None,
            reason="timeout",
            message="Health probe timed out.",
        )
    except httpx.ConnectError:
        return ServiceHealthResult(
            ok=False,
            status_code=None,
            reason="connection_failed",
            message="Health endpoint is not reachable.",
        )
    except httpx.HTTPError:
        return ServiceHealthResult(
            ok=False,
            status_code=None,
            reason="request_failed",
            message="Health probe request failed.",
        )

    status_code = response.status_code
    if 200 <= status_code <= 299:
        return ServiceHealthResult(
            ok=True,
            status_code=status_code,
            reason="ok",
            message="Health endpoint is ready.",
        )
    if status_code in {401, 403}:
        reason = "authentication_rejected"
        message = "Health endpoint rejected authentication."
    elif status_code == 404:
        reason = "endpoint_not_found"
        message = "Configured health endpoint was not found."
    elif 300 <= status_code <= 399:
        reason = "redirect"
        message = "Health endpoint returned a redirect."
    else:
        reason = "http_error"
        message = f"Health endpoint returned HTTP {status_code}."
    return ServiceHealthResult(
        ok=False,
        status_code=status_code,
        reason=reason,
        message=message,
    )
