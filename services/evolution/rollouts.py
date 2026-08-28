"""Compatibility facade for durable shadow/canary rollout safety.

LoRA canaries use an independently addressable immutable candidate Runtime
identity. The service fails closed when that capability is not proven and never
changes a global adapter pointer per request.
"""

# Public rollout types and helpers remain available from this facade.

from services.evolution.rollout_activation import RolloutActivation
from services.evolution.rollout_approvals import RolloutApprovals
from services.evolution.rollout_audit import RolloutAudit
from services.evolution.rollout_canary import RolloutCanary
from services.evolution.rollout_evaluation import (
    RealPromptShadowEvaluator,
    reviewed_dataset_hash,
)
from services.evolution.rollout_inspection import inspect_rollout_state
from services.evolution.rollout_models import (
    CanaryContext,
    CanarySelection,
    CandidateType,
    InvalidRolloutTransition,
    PromotionReadiness,
    RolloutBlocked,
    RolloutError,
    RolloutRecord,
    RolloutState,
    ShadowEvaluator,
    ShadowMetrics,
)
from services.evolution.rollout_persistence import RolloutPersistence
from services.evolution.rollout_reconciliation import RolloutReconciliation
from services.evolution.rollout_shadow import RolloutShadow
from services.evolution.rollout_support import RolloutSupport


class RolloutService(
    RolloutPersistence,
    RolloutShadow,
    RolloutCanary,
    RolloutActivation,
    RolloutReconciliation,
    RolloutSupport,
    RolloutAudit,
    RolloutApprovals,
):
    """Stable manager facade composed from focused rollout services."""


__all__ = [
    "CanaryContext",
    "CanarySelection",
    "CandidateType",
    "InvalidRolloutTransition",
    "PromotionReadiness",
    "RealPromptShadowEvaluator",
    "RolloutBlocked",
    "RolloutError",
    "RolloutRecord",
    "RolloutService",
    "RolloutState",
    "ShadowEvaluator",
    "ShadowMetrics",
    "inspect_rollout_state",
    "reviewed_dataset_hash",
]
