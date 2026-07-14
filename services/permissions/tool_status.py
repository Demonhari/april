from __future__ import annotations

from enum import StrEnum


class ToolCallStatus(StrEnum):
    """Canonical durable tool-call outcomes."""

    EXECUTED = "executed"
    FAILED = "failed"
    DENIED = "denied"
    PENDING_APPROVAL = "pending_approval"
    CANCELLED = "cancelled"


# Reads remain compatible with rows written before schema v14.
HISTORICAL_SUCCESS_STATUSES: tuple[str, ...] = (
    ToolCallStatus.EXECUTED.value,
    "ok",
    "success",
)


def is_successful_tool_call_status(value: object) -> bool:
    return isinstance(value, str) and value in HISTORICAL_SUCCESS_STATUSES
