from __future__ import annotations

from services.permissions.schemas import RISK_ORDER, RiskLevel


def max_risk(*risks: str) -> RiskLevel:
    selected = max(risks, key=lambda risk: RISK_ORDER.get(risk, 5))
    if selected not in RISK_ORDER:
        return "external_action"
    return selected  # type: ignore[return-value]


def level_for_risk(risk: str) -> int:
    return RISK_ORDER.get(risk, 5)
