"""Intelligent underwriting engines for GrabCredit and GrabInsurance."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Literal, TypedDict, cast

from app.explainability import CategoryAverages, generate_underwriting_decision_trail
from app.risk_engine import MerchantLike

Tier = Literal["Tier 1", "Tier 2", "Tier 3"]


class FinancialBreakdown(TypedDict):
    """Reusable calculation breakdown for UI transparency and audit."""

    inputs: dict[str, float | str | bool]
    formula: str
    computed_values: dict[str, float | str | bool]


class GrabCreditOffer(TypedDict):
    mode: Literal["grab_credit"]
    tier: str
    credit_limit_lakhs: float
    interest_rate_pct: float
    interest_rate_band_pct: str
    tenure_options_months: list[int]
    exposure_cap_applied: bool
    haircut_applied: bool
    decision_metrics: dict[str, float | str]
    financial_breakdown: FinancialBreakdown


class GrabInsuranceOffer(TypedDict):
    mode: Literal["grab_insurance"]
    tier: str
    coverage_amount_inr: float
    premium_rate_pct: float
    premium_quote_inr: float
    policy_type: str
    decision_metrics: dict[str, float | str]
    financial_breakdown: FinancialBreakdown


class RejectionOffer(TypedDict):
    mode: str
    status: Literal["REJECTED"]
    rejection_reason: str
    failed_metrics: dict[str, float]
    advisory_note: str
    decision_metrics: dict[str, float | str]


class UnderwritingDecision(TypedDict):
    merchant_id: str
    risk_score: float
    confidence_level: str
    offer: GrabCreditOffer | GrabInsuranceOffer | RejectionOffer


@dataclass(frozen=True)
class CreditPolicyConfig:
    base_credit_multiplier_last3m: float = 0.60
    tier_multiplier: dict[str, float] | None = None
    cap_multiple_avg_monthly_gmv: float = 3.0
    minimum_credit_lakhs: float = 2.0
    exposure_cap_lakhs: float = 10.0
    tier2_refund_haircut_pct: float = 5.0

    tier_interest_band: dict[str, tuple[float, float]] | None = None
    tier_tenure_map: dict[str, list[int]] | None = None

    def __post_init__(self) -> None:
        if self.tier_multiplier is None:
            object.__setattr__(self, "tier_multiplier", {"Tier 1": 1.0, "Tier 2": 0.75, "Tier 3": 0.5})
        if self.tier_interest_band is None:
            object.__setattr__(self, "tier_interest_band", {"Tier 1": (12.0, 14.0), "Tier 2": (15.0, 18.0), "Tier 3": (19.0, 22.0)})
        if self.tier_tenure_map is None:
            object.__setattr__(self, "tier_tenure_map", {"Tier 1": [6, 9, 12], "Tier 2": [6, 9], "Tier 3": [6]})


@dataclass(frozen=True)
class InsurancePolicyConfig:
    tier_coverage_multiple: dict[str, float] | None = None
    base_premium_rate_pct: float = 1.8
    risk_loading_pct: float = 0.5
    volatility_cv_threshold: float = 0.40
    high_seasonality_threshold: float = 1.80

    def __post_init__(self) -> None:
        if self.tier_coverage_multiple is None:
            object.__setattr__(self, "tier_coverage_multiple", {"Tier 1": 2.0, "Tier 2": 1.5, "Tier 3": 1.0})


@dataclass(frozen=True)
class UnderwritingConfig:
    credit: CreditPolicyConfig = CreditPolicyConfig()
    insurance: InsurancePolicyConfig = InsurancePolicyConfig()


DEFAULT_UNDERWRITING_CONFIG = UnderwritingConfig()


def _avg_last_3_months_gmv(merchant: MerchantLike) -> float:
    return mean(merchant.monthly_gmv_12m[-3:])


def _normalize_risk_score(risk_score: float) -> float:
    return max(0.0, min(1.0, risk_score / 100.0))


def _interpolate_interest_rate(min_rate: float, max_rate: float, risk_score: float) -> float:
    normalized = _normalize_risk_score(risk_score)
    return round(min_rate + ((max_rate - min_rate) * normalized), 2)


def _hard_rejection_check(decision_metrics: dict[str, float | str]) -> tuple[bool, str, dict[str, float], str]:
    """Apply hard rejection layer from decision metrics without recomputation."""
    failed: dict[str, float] = {}

    refund_rate = float(decision_metrics["refund_rate_pct"])
    yoy_growth = float(decision_metrics["yoy_gmv_growth_pct"])
    volatility_cv = float(decision_metrics["gmv_coefficient_of_variation"])

    if refund_rate >= 8.0:
        failed["return_and_refund_rate_pct"] = refund_rate
    if yoy_growth <= -10.0:
        failed["yoy_gmv_growth_pct"] = yoy_growth
    if volatility_cv >= 0.6:
        failed["gmv_volatility_coefficient"] = volatility_cv

    if not failed:
        return False, "", {}, ""

    if "return_and_refund_rate_pct" in failed:
        reason = "Rejected due to elevated return/refund risk beyond policy tolerance"
    elif "yoy_gmv_growth_pct" in failed:
        reason = "Rejected due to severe negative YoY GMV trajectory"
    else:
        reason = "Rejected due to excessive revenue volatility"

    advisory = (
        "Improve refund controls, stabilize monthly GMV trends, and reduce revenue volatility over subsequent "
        "review cycles before reapplying for automated offers."
    )
    return True, reason, failed, advisory


def _compute_credit_limit_inr(
    decision_metrics: dict[str, float | str],
    tier: Tier,
    merchant: MerchantLike,
    config: UnderwritingConfig,
    override_triggered: bool,
) -> tuple[float, FinancialBreakdown]:
    avg_monthly_gmv = float(decision_metrics["avg_monthly_gmv_inr"])
    avg_last3 = _avg_last_3_months_gmv(merchant)

    base_credit = avg_last3 * config.credit.base_credit_multiplier_last3m
    tier_multiplier = config.credit.tier_multiplier[tier]
    pre_cap_credit = base_credit * tier_multiplier
    cap_value = avg_monthly_gmv * config.credit.cap_multiple_avg_monthly_gmv
    floor_value = config.credit.minimum_credit_lakhs * 100_000

    if tier == "Tier 3" and override_triggered:
        final_credit = 0.0
    else:
        final_credit = max(floor_value, min(pre_cap_credit, cap_value))

    breakdown: FinancialBreakdown = {
        "inputs": {
            "avg_last_3_months_gmv_inr": round(avg_last3, 2),
            "avg_monthly_gmv_inr": round(avg_monthly_gmv, 2),
            "base_credit_multiplier_last3m": config.credit.base_credit_multiplier_last3m,
            "tier_multiplier": tier_multiplier,
            "cap_multiple_avg_monthly_gmv": config.credit.cap_multiple_avg_monthly_gmv,
            "minimum_credit_lakhs": config.credit.minimum_credit_lakhs,
            "override_triggered": override_triggered,
        },
        "formula": "credit = clamp(avg_last_3m * 0.6 * tier_multiplier, min=₹2L, max=3x avg_monthly_gmv)",
        "computed_values": {
            "base_credit_inr": round(base_credit, 2),
            "pre_cap_credit_inr": round(pre_cap_credit, 2),
            "cap_value_inr": round(cap_value, 2),
            "floor_value_inr": round(floor_value, 2),
            "final_credit_inr": round(final_credit, 2),
        },
    }
    return final_credit, breakdown


def _has_policy_override(primary_risk_drivers: list[str]) -> bool:
    return any("override triggered" in item.lower() for item in primary_risk_drivers)


def _apply_capital_haircut(
    tier: Tier,
    credit_limit_lakhs: float,
    decision_metrics: dict[str, float | str],
    config: UnderwritingConfig,
) -> tuple[float, bool]:
    """Apply Tier-2 haircut when refunds are above category average."""
    category_avg_refund = decision_metrics.get("category_avg_refund_rate_pct")
    if (
        tier == "Tier 2"
        and category_avg_refund is not None
        and float(decision_metrics["refund_rate_pct"]) > float(category_avg_refund)
    ):
        haircut_multiplier = 1.0 - (config.credit.tier2_refund_haircut_pct / 100.0)
        return round(credit_limit_lakhs * haircut_multiplier, 2), True

    return credit_limit_lakhs, False


def _apply_exposure_guard(
    tier: Tier,
    confidence_level: str,
    credit_limit_lakhs: float,
    config: UnderwritingConfig,
) -> tuple[float, bool]:
    """Cap exposure for weak-quality books."""
    if (tier == "Tier 3" or confidence_level == "Low Confidence") and credit_limit_lakhs > config.credit.exposure_cap_lakhs:
        return config.credit.exposure_cap_lakhs, True

    return credit_limit_lakhs, False


def _build_grab_credit_offer(
    merchant: MerchantLike,
    tier: Tier,
    decision_metrics: dict[str, float | str],
    confidence_level: str,
    primary_risk_drivers: list[str],
    config: UnderwritingConfig,
) -> GrabCreditOffer:
    risk_score = float(decision_metrics["risk_score"])
    override_triggered = _has_policy_override(primary_risk_drivers)

    credit_limit_inr, breakdown = _compute_credit_limit_inr(
        decision_metrics=decision_metrics,
        tier=tier,
        merchant=merchant,
        config=config,
        override_triggered=override_triggered,
    )
    credit_limit_lakhs = round(credit_limit_inr / 100_000, 2)

    credit_limit_lakhs, haircut_applied = _apply_capital_haircut(
        tier=tier,
        credit_limit_lakhs=credit_limit_lakhs,
        decision_metrics=decision_metrics,
        config=config,
    )
    credit_limit_lakhs, exposure_cap_applied = _apply_exposure_guard(
        tier=tier,
        confidence_level=confidence_level,
        credit_limit_lakhs=credit_limit_lakhs,
        config=config,
    )

    min_rate, max_rate = config.credit.tier_interest_band[tier]
    interest_rate_pct = _interpolate_interest_rate(min_rate, max_rate, risk_score)

    breakdown["computed_values"]["interest_rate_pct"] = interest_rate_pct
    breakdown["computed_values"]["final_credit_lakhs"] = credit_limit_lakhs
    breakdown["computed_values"]["haircut_applied"] = haircut_applied
    breakdown["computed_values"]["exposure_cap_applied"] = exposure_cap_applied
    breakdown["inputs"]["interest_band_min_pct"] = min_rate
    breakdown["inputs"]["interest_band_max_pct"] = max_rate
    breakdown["inputs"]["risk_score"] = risk_score
    breakdown["inputs"]["confidence_level"] = confidence_level
    breakdown["inputs"]["tier2_refund_haircut_pct"] = config.credit.tier2_refund_haircut_pct
    breakdown["inputs"]["exposure_cap_lakhs"] = config.credit.exposure_cap_lakhs

    return {
        "mode": "grab_credit",
        "tier": tier,
        "credit_limit_lakhs": credit_limit_lakhs,
        "interest_rate_pct": interest_rate_pct,
        "interest_rate_band_pct": f"{min_rate:.1f}-{max_rate:.1f}%",
        "tenure_options_months": config.credit.tier_tenure_map[tier],
        "exposure_cap_applied": exposure_cap_applied,
        "haircut_applied": haircut_applied,
        "decision_metrics": decision_metrics,
        "financial_breakdown": breakdown,
    }


def _determine_policy_type(
    decision_metrics: dict[str, float | str],
    config: UnderwritingConfig,
) -> str:
    cv = float(decision_metrics["gmv_coefficient_of_variation"])
    seasonality = float(decision_metrics["seasonality_index"])

    if seasonality > config.insurance.high_seasonality_threshold:
        return "Seasonal Interruption Cover"
    if cv > config.insurance.volatility_cv_threshold:
        return "Revenue Protection Cover"
    return "Business Stability Cover"


def _build_grab_insurance_offer(
    tier: Tier,
    decision_metrics: dict[str, float | str],
    config: UnderwritingConfig,
) -> GrabInsuranceOffer:
    avg_monthly_gmv = float(decision_metrics["avg_monthly_gmv_inr"])
    coverage_multiple = config.insurance.tier_coverage_multiple[tier]
    coverage_amount = round(avg_monthly_gmv * coverage_multiple, 2)

    premium_rate = config.insurance.base_premium_rate_pct

    category_avg_refund = decision_metrics.get("category_avg_refund_rate_pct")
    refund_rate = float(decision_metrics["refund_rate_pct"])
    if category_avg_refund is not None and refund_rate > float(category_avg_refund):
        premium_rate += config.insurance.risk_loading_pct

    cv = float(decision_metrics["gmv_coefficient_of_variation"])
    if cv > config.insurance.volatility_cv_threshold:
        premium_rate += config.insurance.risk_loading_pct

    premium_quote = round(coverage_amount * premium_rate / 100.0, 2)
    policy_type = _determine_policy_type(decision_metrics, config)

    breakdown: FinancialBreakdown = {
        "inputs": {
            "avg_monthly_gmv_inr": round(avg_monthly_gmv, 2),
            "coverage_multiple": coverage_multiple,
            "base_premium_rate_pct": config.insurance.base_premium_rate_pct,
            "risk_loading_pct": config.insurance.risk_loading_pct,
            "refund_rate_pct": refund_rate,
            "category_avg_refund_rate_pct": float(category_avg_refund) if category_avg_refund is not None else -1.0,
            "gmv_coefficient_of_variation": cv,
            "volatility_cv_threshold": config.insurance.volatility_cv_threshold,
        },
        "formula": "coverage = avg_monthly_gmv * tier_multiple; premium_rate = 1.8% + refund_loading + volatility_loading; premium = coverage * rate",
        "computed_values": {
            "coverage_amount_inr": coverage_amount,
            "final_premium_rate_pct": round(premium_rate, 2),
            "premium_quote_inr": premium_quote,
            "policy_type": policy_type,
        },
    }

    return {
        "mode": "grab_insurance",
        "tier": tier,
        "coverage_amount_inr": coverage_amount,
        "premium_rate_pct": round(premium_rate, 2),
        "premium_quote_inr": premium_quote,
        "policy_type": policy_type,
        "decision_metrics": decision_metrics,
        "financial_breakdown": breakdown,
    }


def _build_rejection_offer(
    mode: Literal["grab_credit", "grab_insurance"],
    reason: str,
    failed_metrics: dict[str, float],
    advisory_note: str,
    decision_metrics: dict[str, float | str],
) -> RejectionOffer:
    return {
        "mode": mode,
        "status": "REJECTED",
        "rejection_reason": reason,
        "failed_metrics": failed_metrics,
        "advisory_note": advisory_note,
        "decision_metrics": decision_metrics,
    }


def generate_underwriting_decision(
    merchant: MerchantLike,
    mode: Literal["grab_credit", "grab_insurance"],
    category_averages: CategoryAverages | None = None,
    config: UnderwritingConfig = DEFAULT_UNDERWRITING_CONFIG,
) -> UnderwritingDecision:
    """Generate structured underwriting decision JSON for a given mode."""
    trail = generate_underwriting_decision_trail(merchant, category_averages=category_averages)
    decision_metrics = trail["decision_metrics"]
    tier = cast(Tier, decision_metrics["tier"])
    risk_score = float(decision_metrics["risk_score"])
    confidence_level = str(trail["confidence_level"])

    rejected, reason, failed_metrics, advisory_note = _hard_rejection_check(decision_metrics)
    if rejected:
        return {
            "merchant_id": str(decision_metrics["merchant_id"]),
            "risk_score": risk_score,
            "confidence_level": confidence_level,
            "offer": _build_rejection_offer(
                mode=mode,
                reason=reason,
                failed_metrics=failed_metrics,
                advisory_note=advisory_note,
                decision_metrics=decision_metrics,
            ),
        }

    if mode == "grab_credit":
        offer = _build_grab_credit_offer(
            merchant=merchant,
            tier=tier,
            decision_metrics=decision_metrics,
            confidence_level=confidence_level,
            primary_risk_drivers=trail["primary_risk_drivers"],
            config=config,
        )
    else:
        offer = _build_grab_insurance_offer(
            tier=tier,
            decision_metrics=decision_metrics,
            config=config,
        )

    return {
        "merchant_id": str(decision_metrics["merchant_id"]),
        "risk_score": risk_score,
        "confidence_level": confidence_level,
        "offer": offer,
    }
