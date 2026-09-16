"""Core data models for Praxos."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Outcome = Literal["success", "failure", "blocked", "unknown"]
Decision = Literal["allow", "warn", "block"]
Severity = Literal["info", "warn", "block"]
Status = Literal["active", "archived"]
ReviewStatus = Literal["pending", "approved", "rejected"]


@dataclass(frozen=True)
class Episode:
    id: str
    workspace_id: str
    agent_id: str
    task: str
    action: str
    outcome: Outcome = "unknown"
    result: str = ""
    human_feedback: str = ""
    source_refs: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    account_id: str = ""
    customer_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceReceipt:
    id: str
    workspace_id: str
    source_uri: str
    snippet: str
    episode_id: str = ""
    observed_at: str = ""
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Lesson:
    id: str
    workspace_id: str
    title: str
    pattern: str
    rule: str
    recommendation: str
    regression_case: str = ""
    policy_candidate: str = ""
    confidence: float = 0.7
    evidence_episode_ids: list[str] = field(default_factory=list)
    evidence_receipt_ids: list[str] = field(default_factory=list)
    status: Status = "active"
    review_status: ReviewStatus = "pending"
    created_at: str = ""
    last_used_at: str | None = None
    use_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Policy:
    id: str
    workspace_id: str
    name: str
    trigger: str
    instruction: str
    severity: Severity = "warn"
    status: Status = "active"
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReviewItem:
    id: str
    workspace_id: str
    item_type: str
    item_id: str
    status: ReviewStatus = "pending"
    summary: str = ""
    created_at: str = ""
    reviewed_at: str | None = None
    reviewer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Account:
    id: str
    workspace_id: str
    name: str
    external_ref: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Customer:
    id: str
    workspace_id: str
    account_id: str
    name: str
    role: str = ""
    external_ref: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Commitment:
    id: str
    workspace_id: str
    account_id: str
    description: str
    source_uri: str = ""
    due_at: str = ""
    status: str = "open"
    confidence: float = 0.8
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Escalation:
    id: str
    workspace_id: str
    account_id: str
    summary: str
    severity: str = "medium"
    status: str = "open"
    source_uri: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    workspace_id: str
    account_id: str
    decision: str
    source_uri: str = ""
    decided_at: str = ""
    status: str = "active"
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ActionCheck:
    id: str
    workspace_id: str
    task: str
    action: str
    decision: Decision
    reasons: list[str] = field(default_factory=list)
    matched_lesson_ids: list[str] = field(default_factory=list)
    matched_policy_ids: list[str] = field(default_factory=list)
    matched_evidence_ids: list[str] = field(default_factory=list)
    matched_business_ids: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
