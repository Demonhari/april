from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AprilError(Exception):
    code: str
    message: str
    status_code: int = 400
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class ConfigError(AprilError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("CONFIG_ERROR", message, 500, details or {})


class RuntimeUnavailableError(AprilError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("RUNTIME_UNAVAILABLE", message, 503, details or {})


class ModelUnavailableError(AprilError):
    def __init__(self, model_id: str, message: str, details: dict[str, Any] | None = None) -> None:
        data = {"model_id": model_id}
        data.update(details or {})
        super().__init__("MODEL_UNAVAILABLE", message, 503, data)


class PermissionDeniedError(AprilError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("PERMISSION_DENIED", message, 403, details or {})


class ApprovalRequiredError(AprilError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("APPROVAL_REQUIRED", message, 403, details or {})


class NotFoundError(AprilError):
    def __init__(self, resource: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("NOT_FOUND", f"{resource} was not found.", 404, details or {})


class ValidationError(AprilError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("VALIDATION_ERROR", message, 422, details or {})


class RequestTooLargeError(AprilError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("REQUEST_TOO_LARGE", message, 413, details or {})


def error_payload(error: AprilError, request_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "code": error.code,
            "message": error.message,
        }
    }
    if request_id:
        payload["error"]["request_id"] = request_id
    if error.details:
        payload["error"]["details"] = error.details
    return payload
