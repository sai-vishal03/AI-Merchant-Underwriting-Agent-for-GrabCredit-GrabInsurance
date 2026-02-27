"""Regulator-safe explainability utilities for underwriting decisions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Literal, TypedDict, cast

from app.llm import generate_llm_response
from app.risk_engine import (
    DEFAULT_CONFIG,
    MerchantLike,
    evaluate_merchant_risk,
    evaluate_merchant_risk_detailed,
)

Tier = Literal["Tier 1", "Tier 2", "Tier 3"]


class DecisionTrail(TypedDict):
    """Structured internal decision trail for dashboard transparency."""

    decision_metrics: dict[str, float | str]
    primary_risk_drivers: list[str]
    secondary_risk_flags: list[str]
    primary_strengths: list[str]
    secondary_strengths: list[str]
    confidence_score: float
    confidence_level: str
    final_explanation: str


@dataclass(frozen=True)
class CategoryAverages:
    refund_rate: float
    customer_return_rate: float
    yoy_gmv_growth: float


@dataclass(frozen=True)
class ExplainabilityConfig:
    refund_rejection_threshold: float = 6.0
    growth_rejection_threshold: float = 0.0
    primary_impact_share_threshold_pct: float = 25.0
    override_confidence_penalty: float = 12.0


DEFAULT_EXPLAINABILITY_CONFIG = ExplainabilityConfig()


def _safe_yoy_growth(merchant: MerchantLike) -> float:
    try:
        return merchant.compute_yoy_gmv_growth()
    except ValueError:
        return 0.0


def _confidence_band(score: float) -> str:
    if score >= 75:
        return "High Confidence"
    if score >= 50:
        return "Moderate Confidence"
    return "Low Confidence"


def _compute_confidence(risk_score: float, override_triggered: bool, config: ExplainabilityConfig) -> tuple[float, str]:
    confidence_score = 100.0 - risk_score
    if override_triggered:
        confidence_score -= config.override_confidence_penalty
    confidence_score = max(0.0, min(100.0, round(confidence_score, 2)))
    return confidence_score, _confidence_band(confidence_score)


def _metric_snapshot_paragraph(merchant: MerchantLike, tier: Tier, growth_pct: float) -> str:
    avg_monthly_gmv = mean(merchant.monthly_gmv_12m)
    return (
        f"Underwriting assessment for merchant {merchant.merchant_id} ({merchant.category}) places the account in {tier}. "
        f"Observed YoY GMV growth is {growth_pct:.1f}%, trailing refund rate is {merchant.return_and_refund_rate:.1f}%, "
        f"customer return rate is {merchant.customer_return_rate:.1f}%, and average monthly GMV is ₹{avg_monthly_gmv:,.0f}."
    )


def _benchmark_paragraph(merchant: MerchantLike, growth_pct: float, category_averages: CategoryAverages | None) -> str:
    if category_averages is None:
        return (
            "Benchmark note: category comparators are unavailable for this run, so the decision is calibrated "
            "against internal policy thresholds and weighted risk scoring."
        )

    refund_delta = merchant.return_and_refund_rate - category_averages.refund_rate
    return_delta = merchant.customer_return_rate - category_averages.customer_return_rate
    growth_delta = growth_pct - category_averages.yoy_gmv_growth
    return (
        f"Benchmark comparison: growth is {abs(growth_delta):.1f} pp {'above' if growth_delta >= 0 else 'below'} "
        f"category ({category_averages.yoy_gmv_growth:.1f}%), refunds are {abs(refund_delta):.1f} pp "
        f"{'above' if refund_delta > 0 else 'below'} category ({category_averages.refund_rate:.1f}%), and "
        f"customer return rate is {abs(return_delta):.1f} pp {'above' if return_delta >= 0 else 'below'} "
        f"category ({category_averages.customer_return_rate:.1f}%)."
    )


def _build_decision_metrics(
    merchant: MerchantLike,
    tier: Tier,
    risk_score: float,
    growth_pct: float,
    confidence_score: float,
    confidence_level: str,
    category_averages: CategoryAverages | None,
) -> dict[str, float | str]:
    avg_monthly_gmv = mean(merchant.monthly_gmv_12m)
    volatility = merchant.compute_gmv_volatility()
    coeff_var = (volatility / avg_monthly_gmv) if avg_monthly_gmv > 0 else 0.0

    metrics: dict[str, float | str] = {
        "merchant_id": merchant.merchant_id,
        "category": merchant.category,
        "tier": tier,
        "risk_score": round(risk_score, 2),
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "yoy_gmv_growth_pct": round(growth_pct, 2),
        "refund_rate_pct": round(merchant.return_and_refund_rate, 2),
        "customer_return_rate_pct": round(merchant.customer_return_rate, 2),
        "avg_monthly_gmv_inr": round(avg_monthly_gmv, 2),
        "gmv_volatility_inr": round(volatility, 2),
        "gmv_coefficient_of_variation": round(coeff_var, 4),
        "seasonality_index": round(merchant.seasonality_index, 3),
    }

    if category_averages is not None:
        metrics.update(
            {
                "category_avg_refund_rate_pct": round(category_averages.refund_rate, 2),
                "category_avg_customer_return_rate_pct": round(category_averages.customer_return_rate, 2),
                "category_avg_yoy_gmv_growth_pct": round(category_averages.yoy_gmv_growth, 2),
            }
        )

    return metrics


def _attribution_from_diagnostics(
    merchant: MerchantLike,
    diagnostics: dict[str, object],
    config: ExplainabilityConfig,
) -> tuple[list[str], list[str], list[str], list[str]]:
    metrics = cast(dict[str, float], diagnostics["metrics"])
    shares = cast(dict[str, float], diagnostics["weighted_contribution_share_pct"])
    overrides = cast(list[str], diagnostics["policy_overrides"])

    risk_map = {
        "refund_risk": f"Refund rate contributes materially to risk ({metrics['refund_rate_pct']:.1f}%; impact share {shares.get('refund_risk', 0.0):.1f}%)",
        "volatility_risk": f"Sales volatility contributes materially to risk (CV-based impact share {shares.get('volatility_risk', 0.0):.1f}%)",
        "growth_risk": f"Growth trajectory contributes to risk (YoY GMV {metrics['gmv_growth_pct']:.1f}%; impact share {shares.get('growth_risk', 0.0):.1f}%)",
        "loyalty_risk": f"Customer loyalty contributes to risk (return rate {metrics['customer_return_rate_pct']:.1f}%; impact share {shares.get('loyalty_risk', 0.0):.1f}%)",
    }

    primary_risk_drivers = [*overrides]
    secondary_risk_flags: list[str] = []

    for key, text in risk_map.items():
        if shares.get(key, 0.0) > config.primary_impact_share_threshold_pct:
            primary_risk_drivers.append(text)
        else:
            secondary_risk_flags.append(text)

    weights = DEFAULT_CONFIG.weights
    protection_scores = {
        key: (100.0 - metrics[key]) * weights[key]
        for key in ("refund_risk", "volatility_risk", "growth_risk", "loyalty_risk")
    }
    total_protection = sum(protection_scores.values()) or 1.0

    strength_map = {
        "refund_risk": f"Refund control supports portfolio quality ({merchant.return_and_refund_rate:.1f}%)",
        "volatility_risk": "Cash-flow variability remains manageable on weighted volatility assessment",
        "growth_risk": f"Growth profile supports credit absorption (YoY GMV {metrics['gmv_growth_pct']:.1f}%)",
        "loyalty_risk": f"Customer stickiness supports revenue durability (return rate {merchant.customer_return_rate:.1f}%)",
    }

    primary_strengths: list[str] = []
    secondary_strengths: list[str] = []
    for key, val in protection_scores.items():
        share = (val / total_protection) * 100.0
        if share > config.primary_impact_share_threshold_pct:
            primary_strengths.append(f"{strength_map[key]} (strength share {share:.1f}%)")
        else:
            secondary_strengths.append(f"{strength_map[key]} (strength share {share:.1f}%)")

    return primary_risk_drivers, secondary_risk_flags, primary_strengths, secondary_strengths


def _explanation_risk_strength_paragraph(
    primary_risk_drivers: list[str],
    primary_strengths: list[str],
    confidence_level: str,
) -> str:
    risk_ref = primary_risk_drivers[0] if primary_risk_drivers else "No dominant primary risk driver was flagged"
    strength_ref = primary_strengths[0] if primary_strengths else "No dominant primary strength was identified"
    return (
        f"Driver summary: primary risk driver noted is {risk_ref}. "
        f"Primary strength noted is {strength_ref}. Confidence level for this underwriting view is {confidence_level}."
    )


def _assert_numeric_traceability(explanation: str, required_tokens: list[str]) -> None:
    if not any(ch.isdigit() for ch in explanation):
        raise ValueError("Explanation must include concrete numeric evidence")
    for token in required_tokens:
        if token not in explanation:
            raise ValueError(f"Explanation missing required metric reference: {token}")


def generate_underwriting_explanation(
    merchant: MerchantLike,
    tier: Tier | None = None,
    risk_score: float | None = None,
    category_averages: CategoryAverages | None = None,
    config: ExplainabilityConfig = DEFAULT_EXPLAINABILITY_CONFIG,
    primary_risk_drivers: list[str] | None = None,
    primary_strengths: list[str] | None = None,
    confidence_level: str | None = None,
) -> str:
    risk_result = evaluate_merchant_risk(merchant)
    diagnostics = evaluate_merchant_risk_detailed(merchant)

    resolved_tier = cast(Tier, tier or risk_result["tier"])
    resolved_risk_score = risk_score if risk_score is not None else risk_result["risk_score"]
    override_triggered = bool(diagnostics["policy_overrides"])
    _, resolved_confidence_level = _compute_confidence(resolved_risk_score, override_triggered, config)
    resolved_confidence_level = confidence_level or resolved_confidence_level

    growth_pct = _safe_yoy_growth(merchant)

    paragraphs = [
        _metric_snapshot_paragraph(merchant, resolved_tier, growth_pct),
        _benchmark_paragraph(merchant, growth_pct, category_averages),
        _explanation_risk_strength_paragraph(
            primary_risk_drivers or [],
            primary_strengths or [],
            resolved_confidence_level,
        ),
        (
            f"Underwriting confidence statement: {resolved_confidence_level} based on model risk score "
            f"{resolved_risk_score:.2f}. This decision remains subject to compliance, KYC, fraud, and legal checks."
        ),
    ]

    explanation = "\n\n".join(paragraphs)
    _assert_numeric_traceability(
        explanation,
        required_tokens=[
            "YoY GMV growth",
            "refund rate",
            "customer return rate",
            resolved_tier,
            resolved_confidence_level,
        ],
    )
    return explanation


def _generate_ai_underwriting_explanation(
    merchant: MerchantLike,
    decision_metrics: dict[str, float | str],
    primary_risk_drivers: list[str],
    primary_strengths: list[str],
    confidence_level: str,
    mode: Literal["grab_credit", "grab_insurance"] = "grab_credit",
) -> str:
    """Generate AI rationale with deterministic fallback if LLM request fails."""
    system_prompt = "You are a senior NBFC credit risk analyst generating underwriting rationales."

    mode_specific_instructions = (
        "- For credit mode, justify limit sizing and rate suitability."
        if mode == "grab_credit"
        else "- For insurance mode, explain coverage amount logic, premium factors, and policy type fit."
    )

    user_prompt = (
        "Generate a professional underwriting rationale using the provided metrics.\n"
        "Requirements:\n"
        "- Write 3-5 sentences.\n"
        "- Reference specific numbers from input.\n"
        "- Avoid generic phrases.\n"
        "- Sound professional and decision-oriented.\n"
        "- Explicitly justify the assigned tier.\n"
        f"{mode_specific_instructions}\n\n"
        f"Mode: {mode}\n"
        f"Merchant ID: {merchant.merchant_id}\n"
        f"Category: {merchant.category}\n"
        f"Tier: {decision_metrics.get('tier')}\n"
        f"Risk Score: {decision_metrics.get('risk_score')}\n"
        f"YoY GMV Growth (%): {decision_metrics.get('yoy_gmv_growth_pct')}\n"
        f"Refund Rate (%): {decision_metrics.get('refund_rate_pct')}\n"
        f"Customer Return Rate (%): {decision_metrics.get('customer_return_rate_pct')}\n"
        f"GMV Volatility (INR): {decision_metrics.get('gmv_volatility_inr')}\n"
        f"GMV Coefficient of Variation: {decision_metrics.get('gmv_coefficient_of_variation')}\n"
        f"Seasonality Index: {decision_metrics.get('seasonality_index')}\n"
        f"Primary Risk Drivers: {primary_risk_drivers}\n"
        f"Primary Strengths: {primary_strengths}\n"
        f"Confidence Level: {confidence_level}\n"
    )

    try:
        ai_text = generate_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
        )
        _assert_numeric_traceability(
            ai_text,
            required_tokens=[
                str(decision_metrics.get("tier")),
                confidence_level,
            ],
        )
        return ai_text
    except Exception:
        return generate_underwriting_explanation(
            merchant=merchant,
            tier=cast(Tier, decision_metrics.get("tier")),
            risk_score=float(decision_metrics.get("risk_score", 0.0)),
            category_averages=None,
            primary_risk_drivers=primary_risk_drivers,
            primary_strengths=primary_strengths,
            confidence_level=confidence_level,
        )


def generate_underwriting_decision_trail(
    merchant: MerchantLike,
    tier: Tier | None = None,
    risk_score: float | None = None,
    category_averages: CategoryAverages | None = None,
    config: ExplainabilityConfig = DEFAULT_EXPLAINABILITY_CONFIG,
    mode: Literal["grab_credit", "grab_insurance"] = "grab_credit",
) -> DecisionTrail:
    """Return structured decision trail for dashboard transparency and UX clarity."""
    risk_result = evaluate_merchant_risk(merchant)
    diagnostics = evaluate_merchant_risk_detailed(merchant)

    resolved_tier = cast(Tier, tier or risk_result["tier"])
    resolved_risk_score = risk_score if risk_score is not None else risk_result["risk_score"]

    override_triggered = bool(diagnostics["policy_overrides"])
    confidence_score, confidence_level = _compute_confidence(
        resolved_risk_score,
        override_triggered,
        config,
    )

    primary_risk_drivers, secondary_risk_flags, primary_strengths, secondary_strengths = _attribution_from_diagnostics(
        merchant,
        diagnostics,
        config,
    )

    growth_pct = _safe_yoy_growth(merchant)
    decision_metrics = _build_decision_metrics(
        merchant=merchant,
        tier=resolved_tier,
        risk_score=resolved_risk_score,
        growth_pct=growth_pct,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        category_averages=category_averages,
    )

    final_explanation = _generate_ai_underwriting_explanation(
        merchant=merchant,
        decision_metrics=decision_metrics,
        primary_risk_drivers=primary_risk_drivers,
        primary_strengths=primary_strengths,
        confidence_level=confidence_level,
        mode=mode,
    )

    return {
        "decision_metrics": decision_metrics,
        "primary_risk_drivers": primary_risk_drivers,
        "secondary_risk_flags": secondary_risk_flags,
        "primary_strengths": primary_strengths,
        "secondary_strengths": secondary_strengths,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "final_explanation": final_explanation,
    }
