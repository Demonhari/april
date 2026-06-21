from __future__ import annotations

import re
from dataclasses import dataclass

SENSITIVE_TERMS = {"password", "secret", "token", "api key", "private key", "credential"}
SENSITIVE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


@dataclass(frozen=True, slots=True)
class MemoryPolicyDecision:
    allowed: bool
    reason: str


class MemoryPolicy:
    def is_sensitive(self, content: str) -> bool:
        normalized = content.lower()
        return any(term in normalized for term in SENSITIVE_TERMS) or any(
            pattern.search(content) for pattern in SENSITIVE_PATTERNS
        )

    def evaluate(self, content: str, *, requested_by_user: bool = False) -> MemoryPolicyDecision:
        if self.is_sensitive(content):
            return MemoryPolicyDecision(
                False, "Sensitive-looking content is not stored as durable memory."
            )
        if requested_by_user:
            return MemoryPolicyDecision(True, "User explicitly requested durable local memory.")
        return MemoryPolicyDecision(
            False, "Conversation messages are not automatically promoted to memory."
        )
