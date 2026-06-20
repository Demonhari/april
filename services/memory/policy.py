from __future__ import annotations

from dataclasses import dataclass

SENSITIVE_TERMS = {"password", "secret", "token", "api key", "private key", "credential"}


@dataclass(frozen=True, slots=True)
class MemoryPolicyDecision:
    allowed: bool
    reason: str


class MemoryPolicy:
    def is_sensitive(self, content: str) -> bool:
        normalized = content.lower()
        return any(term in normalized for term in SENSITIVE_TERMS)

    def evaluate(self, content: str, *, requested_by_user: bool = False) -> MemoryPolicyDecision:
        normalized = content.lower()
        if self.is_sensitive(content):
            return MemoryPolicyDecision(
                False, "Sensitive-looking content is not stored as durable memory."
            )
        if requested_by_user:
            return MemoryPolicyDecision(True, "User explicitly requested durable local memory.")
        if any(phrase in normalized for phrase in ("i prefer", "remember that", "my project")):
            return MemoryPolicyDecision(
                True, "Content appears to be a durable preference or project fact."
            )
        return MemoryPolicyDecision(
            False, "Conversation messages are not automatically promoted to memory."
        )
