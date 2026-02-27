"""Portfolio-level analytics for underwriting decisions.

This module derives aggregate portfolio metrics from existing underwriting
outputs without re-running risk or underwriting logic.
"""

from __future__ import annotations

from typing import Any


CONFIDENCE_BUCKETS: tuple[str, str, str] = (
    "High Confidence",
    "Moderate Confidence",
    "Low Confidence",
)

TIER_BUCKETS: tuple[str, str, str] = ("Tier 1", "Tier 2", "Tier 3")


def _is_rejected(decision: dict[str, Any]) -> bool:
    """Return True when decision offer is explicitly marked as rejected."""
    offer = decision.get("offer", {})
    return isinstance(offer, dict) and offer.get("status") == "REJECTED"


def _extract_tier(decision: dict[str, Any]) -> str | None:
    """Extract tier from offer when available and valid."""
    offer = decision.get("offer", {})
    if not isinstance(offer, dict):
        return None
    tier = offer.get("tier")
    if isinstance(tier, str) and tier in TIER_BUCKETS:
        return tier
    return None


def _extract_confidence(decision: dict[str, Any]) -> str | None:
    """Extract normalized confidence bucket from decision-level metadata."""
    value = decision.get("confidence_level")
    if isinstance(value, str) and value in CONFIDENCE_BUCKETS:
        return value
    return None


def _extract_confidence_score(decision: dict[str, Any]) -> float | None:
    """Extract confidence score from decision metrics if available.

    Decision outputs currently store the numeric score inside
    `offer.decision_metrics.confidence_score`.
    """
    offer = decision.get("offer", {})
    if not isinstance(offer, dict):
        return None

    decision_metrics = offer.get("decision_metrics", {})
    if not isinstance(decision_metrics, dict):
        return None

    raw_score = decision_metrics.get("confidence_score")
    if isinstance(raw_score, (int, float)):
        return float(raw_score)

    return None


def _extract_credit_exposure_lakhs(decision: dict[str, Any]) -> float:
    """Return approved credit exposure in lakhs, else 0.0."""
    if _is_rejected(decision):
        return 0.0

    offer = decision.get("offer", {})
    if not isinstance(offer, dict):
        return 0.0

    if offer.get("mode") != "grab_credit":
        return 0.0

    credit_limit_lakhs = offer.get("credit_limit_lakhs")
    if isinstance(credit_limit_lakhs, (int, float)):
        return float(credit_limit_lakhs)

    return 0.0


def generate_portfolio_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate aggregate portfolio summary from underwriting decisions.

    Args:
        decisions: List of decision objects returned by
            `generate_underwriting_decision(...)`.

    Returns:
        Dictionary with portfolio KPIs and distributions.
    """
    total_merchants = len(decisions)
    total_rejected = sum(1 for decision in decisions if _is_rejected(decision))
    total_approved = total_merchants - total_rejected

    rejection_rate_percentage = (
        round((total_rejected / total_merchants) * 100.0, 2) if total_merchants else 0.0
    )

    tier_distribution: dict[str, int] = {tier: 0 for tier in TIER_BUCKETS}
    confidence_distribution: dict[str, int] = {bucket: 0 for bucket in CONFIDENCE_BUCKETS}

    credit_exposure_total_lakhs = 0.0
    confidence_scores: list[float] = []

    for decision in decisions:
        tier = _extract_tier(decision)
        if tier is not None:
            tier_distribution[tier] += 1

        confidence_bucket = _extract_confidence(decision)
        if confidence_bucket is not None:
            confidence_distribution[confidence_bucket] += 1

        confidence_score = _extract_confidence_score(decision)
        if confidence_score is not None:
            confidence_scores.append(confidence_score)

        credit_exposure_total_lakhs += _extract_credit_exposure_lakhs(decision)

    avg_confidence_score = (
        round(sum(confidence_scores) / len(confidence_scores), 2)
        if confidence_scores
        else 0.0
    )

    return {
        "total_merchants": total_merchants,
        "total_approved": total_approved,
        "total_rejected": total_rejected,
        "rejection_rate_percentage": rejection_rate_percentage,
        "tier_distribution": tier_distribution,
        "total_credit_exposure_lakhs": round(credit_exposure_total_lakhs, 2),
        "avg_confidence_score": avg_confidence_score,
        "confidence_distribution": confidence_distribution,
    }
