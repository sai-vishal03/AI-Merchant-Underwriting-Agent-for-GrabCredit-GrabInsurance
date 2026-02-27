"""Scalable risk evaluation engine for merchant underwriting.

Provides both standard tier/risk outputs and detailed diagnostics used by
explainability and dashboard transparency layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Protocol, TypedDict


class MerchantLike(Protocol):
    """Protocol for merchant objects accepted by the risk engine."""

    merchant_id: str
    category: str
    monthly_gmv_12m: list[float]
    customer_return_rate: float
    seasonality_index: float
    deal_exclusivity_rate: float
    coupon_redemption_rate: float
    return_and_refund_rate: float

    def compute_yoy_gmv_growth(self) -> float: ...

    def compute_gmv_volatility(self) -> float: ...


class RiskEvaluationResult(TypedDict):
    """Standardized response payload for underwriting output."""

    tier: str
    risk_score: float
    decision_recommendation: str


class RiskDiagnostics(TypedDict):
    """Detailed risk diagnostics for auditability and factor attribution."""

    metrics: dict[str, float]
    weighted_contributions: dict[str, float]
    weighted_contribution_share_pct: dict[str, float]
    policy_overrides: list[str]
    risk_score: float


@dataclass(frozen=True)
class TierPolicy:
    tier1_growth_min: float = 20.0
    tier1_refund_max: float = 3.0
    tier1_return_rate_min: float = 60.0
    tier3_refund_min: float = 6.0
    tier3_growth_max: float = 0.0


@dataclass(frozen=True)
class ScoreBands:
    tier1_max_risk_score: float = 35.0
    tier2_max_risk_score: float = 65.0


@dataclass(frozen=True)
class RiskEngineConfig:
    policy: TierPolicy = TierPolicy()
    bands: ScoreBands = ScoreBands()
    weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            object.__setattr__(
                self,
                "weights",
                {
                    "refund_risk": 0.32,
                    "volatility_risk": 0.24,
                    "growth_risk": 0.24,
                    "loyalty_risk": 0.20,
                },
            )


DEFAULT_CONFIG = RiskEngineConfig()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_yoy_growth_pct(merchant: MerchantLike) -> float:
    try:
        return merchant.compute_yoy_gmv_growth()
    except ValueError:
        return 0.0


def _proportional_refund_risk(refund_rate_pct: float) -> float:
    return round(_clamp((refund_rate_pct / 10.0) * 100.0, 0.0, 100.0), 2)


def _proportional_volatility_risk(merchant: MerchantLike) -> float:
    avg_gmv = mean(merchant.monthly_gmv_12m)
    if avg_gmv <= 0:
        return 100.0

    volatility = merchant.compute_gmv_volatility()
    coeff_var = volatility / avg_gmv
    monthly_declines = sum(
        1
        for previous, current in zip(merchant.monthly_gmv_12m, merchant.monthly_gmv_12m[1:])
        if current < previous
    )
    decline_ratio = monthly_declines / 11.0

    cv_risk = _clamp((coeff_var / 0.80) * 100.0, 0.0, 100.0)
    decline_risk = _clamp(decline_ratio * 100.0, 0.0, 100.0)
    return round((0.75 * cv_risk) + (0.25 * decline_risk), 2)


def _proportional_growth_risk(growth_pct: float) -> float:
    normalized = (growth_pct + 20.0) / 60.0
    return round(100.0 - _clamp(normalized * 100.0, 0.0, 100.0), 2)


def _proportional_loyalty_risk(customer_return_rate_pct: float) -> float:
    return round(100.0 - _clamp(customer_return_rate_pct, 0.0, 100.0), 2)


def _collect_metrics(merchant: MerchantLike) -> dict[str, float]:
    growth_pct = _safe_yoy_growth_pct(merchant)
    return {
        "gmv_growth_pct": growth_pct,
        "refund_rate_pct": merchant.return_and_refund_rate,
        "customer_return_rate_pct": merchant.customer_return_rate,
        "refund_risk": _proportional_refund_risk(merchant.return_and_refund_rate),
        "volatility_risk": _proportional_volatility_risk(merchant),
        "growth_risk": _proportional_growth_risk(growth_pct),
        "loyalty_risk": _proportional_loyalty_risk(merchant.customer_return_rate),
    }


def _weighted_risk_score(metrics: dict[str, float], config: RiskEngineConfig) -> tuple[float, dict[str, float], dict[str, float]]:
    weighted: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for metric_name, weight in config.weights.items():
        contribution = metrics[metric_name] * weight
        weighted[metric_name] = round(contribution, 4)
        weighted_sum += contribution
        total_weight += weight

    if total_weight <= 0:
        return 100.0, weighted, {name: 0.0 for name in weighted}

    risk_score = round(_clamp(weighted_sum / total_weight, 0.0, 100.0), 2)
    share_pct: dict[str, float] = {}
    denom = weighted_sum if weighted_sum > 0 else 1.0
    for name, contribution in weighted.items():
        share_pct[name] = round((contribution / denom) * 100.0, 2)

    return risk_score, weighted, share_pct


def _policy_overrides(metrics: dict[str, float], config: RiskEngineConfig) -> list[str]:
    overrides: list[str] = []
    if metrics["refund_rate_pct"] >= config.policy.tier3_refund_min:
        overrides.append("Refund policy override triggered (refund rate >= 6%)")
    if metrics["gmv_growth_pct"] < config.policy.tier3_growth_max:
        overrides.append("Growth policy override triggered (negative YoY GMV growth)")
    return overrides


def _classify_tier(metrics: dict[str, float], risk_score: float, config: RiskEngineConfig) -> str:
    overrides = _policy_overrides(metrics, config)
    if overrides:
        return "Tier 3"

    if risk_score <= config.bands.tier1_max_risk_score:
        if (
            metrics["gmv_growth_pct"] > config.policy.tier1_growth_min
            and metrics["refund_rate_pct"] < config.policy.tier1_refund_max
            and metrics["customer_return_rate_pct"] > config.policy.tier1_return_rate_min
        ):
            return "Tier 1"
        return "Tier 2"

    if risk_score <= config.bands.tier2_max_risk_score:
        return "Tier 2"

    return "Tier 3"


def _decision_recommendation_for_tier(tier: str) -> str:
    if tier == "Tier 1":
        return "Pre-approve with best rates and expedited disbursal controls."
    if tier == "Tier 2":
        return "Pre-approve with standard pricing and stricter exposure caps."
    return "Manual underwriter review, additional documents, or decline."


def evaluate_merchant_risk_detailed(
    merchant: MerchantLike, config: RiskEngineConfig = DEFAULT_CONFIG
) -> RiskDiagnostics:
    """Return detailed risk diagnostics for attribution and transparency."""
    metrics = _collect_metrics(merchant)
    risk_score, weighted_contributions, contribution_share = _weighted_risk_score(metrics, config)
    overrides = _policy_overrides(metrics, config)
    return {
        "metrics": metrics,
        "weighted_contributions": weighted_contributions,
        "weighted_contribution_share_pct": contribution_share,
        "policy_overrides": overrides,
        "risk_score": risk_score,
    }


def evaluate_merchant_risk(
    merchant: MerchantLike, config: RiskEngineConfig = DEFAULT_CONFIG
) -> RiskEvaluationResult:
    diagnostics = evaluate_merchant_risk_detailed(merchant, config)
    tier = _classify_tier(diagnostics["metrics"], diagnostics["risk_score"], config)
    return {
        "tier": tier,
        "risk_score": diagnostics["risk_score"],
        "decision_recommendation": _decision_recommendation_for_tier(tier),
    }
