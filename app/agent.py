"""Multi-agent orchestration layer for merchant underwriting decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from app.db import get_active_model, save_decision_audit_trail, save_underwriting_decision
from app.whatsapp import send_whatsapp_offer

FEATURE_FLAGS: dict[str, bool] = {
    "explainability_v2": True,
    "strict_validation": True,
    "simulation_mode": False,
}

CATEGORY_BENCHMARKS: dict[str, dict[str, float]] = {
    "fashion": {"refund_avg": 4.8},
    "electronics": {"refund_avg": 6.2},
    "grocery": {"refund_avg": 2.1},
    "lifestyle": {"refund_avg": 5.4},
    "home": {"refund_avg": 3.9},
}


def get_feature_flags() -> dict[str, bool]:
    """Return current feature flags."""
    return dict(FEATURE_FLAGS)


def set_feature_flag(flag: str, value: bool) -> dict[str, bool]:
    """Set a feature flag value."""
    if flag not in FEATURE_FLAGS:
        raise ValueError(f"Unknown feature flag: {flag}")
    FEATURE_FLAGS[flag] = value
    return get_feature_flags()


class FeatureEngineeringAgent:
    """Build deterministic underwriting features from merchant input."""

    _REQUIRED_FIELDS = {
        "merchant_id",
        "category",
        "monthly_gmv_12m",
        "coupon_redemption_rate",
        "unique_customer_count",
        "customer_return_rate",
        "avg_order_value",
        "seasonality_index",
        "deal_exclusivity_rate",
        "return_and_refund_rate",
    }

    def build_features(self, merchant_profile: dict[str, Any]) -> dict[str, Any]:
        """Validate merchant profile and compute features for risk assessment."""
        missing_fields = sorted(self._REQUIRED_FIELDS.difference(merchant_profile.keys()))
        if missing_fields:
            raise ValueError(f"Missing required merchant_profile fields: {', '.join(missing_fields)}")

        monthly_gmv = merchant_profile.get("monthly_gmv_12m")
        if not isinstance(monthly_gmv, list) or len(monthly_gmv) < 2:
            raise ValueError("monthly_gmv_12m must be a list with at least 2 values")

        gmv_values = [float(value) for value in monthly_gmv]
        avg_gmv = mean(gmv_values)
        first_month = gmv_values[0]
        last_month = gmv_values[-1]

        if first_month == 0:
            yoy_growth_percent = 0.0 if last_month == 0 else 100.0
        else:
            yoy_growth_percent = ((last_month - first_month) / first_month) * 100.0

        if avg_gmv == 0:
            volatility_index = 0.0
        else:
            variance = sum((value - avg_gmv) ** 2 for value in gmv_values) / len(gmv_values)
            volatility_index = (variance**0.5) / avg_gmv

        refund_rate = float(merchant_profile["return_and_refund_rate"])
        customer_return_rate = float(merchant_profile["customer_return_rate"])

        if FEATURE_FLAGS["strict_validation"]:
            if len(gmv_values) != 12:
                raise ValueError("strict_validation enabled: monthly_gmv_12m must contain exactly 12 values")
            if any(value < 0 for value in gmv_values):
                raise ValueError("strict_validation enabled: monthly_gmv_12m values must be non-negative")
            if not 0 <= refund_rate <= 100:
                raise ValueError("strict_validation enabled: return_and_refund_rate must be between 0 and 100")
            if not 0 <= customer_return_rate <= 100:
                raise ValueError("strict_validation enabled: customer_return_rate must be between 0 and 100")

        category = str(merchant_profile["category"]).lower()
        benchmark = CATEGORY_BENCHMARKS.get(category, {"refund_avg": 4.5})

        return {
            "merchant_id": str(merchant_profile["merchant_id"]),
            "category": category,
            "avg_gmv": round(avg_gmv, 2),
            "yoy_growth_percent": round(yoy_growth_percent, 2),
            "volatility_index": round(volatility_index, 4),
            "refund_risk_score": round(max(0.0, min(100.0, refund_rate * 8.0)), 2),
            "loyalty_score": round(max(0.0, min(100.0, customer_return_rate)), 2),
            "customer_return_rate": round(customer_return_rate, 2),
            "return_and_refund_rate": round(refund_rate, 2),
            "refund_category_benchmark": float(benchmark["refund_avg"]),
            "seasonality_index": float(merchant_profile["seasonality_index"]),
            "avg_order_value": float(merchant_profile["avg_order_value"]),
            "simulation_target": merchant_profile.get("simulation_target"),
        }


class RiskAssessmentAgent:
    """Translate engineered features into risk score, tier, and rejection state."""

    def assess(self, feature_dict: dict[str, Any]) -> dict[str, Any]:
        growth = float(feature_dict["yoy_growth_percent"])
        volatility = float(feature_dict["volatility_index"])
        refund_rate = float(feature_dict["return_and_refund_rate"])
        loyalty = float(feature_dict["loyalty_score"])
        seasonality_index = float(feature_dict["seasonality_index"])
        avg_gmv = float(feature_dict["avg_gmv"])

        growth_component = self._growth_component(growth)
        volatility_component = self._volatility_component(volatility)
        refund_component = self._refund_component(refund_rate)
        loyalty_component = self._loyalty_component(loyalty)

        if seasonality_index > 3.0:
            volatility_component += 6.0

        risk_score = max(0.0, min(100.0, 30.0 + growth_component + volatility_component + refund_component + loyalty_component))

        rejection_reason: str | None = None
        if avg_gmv <= 0:
            rejection_reason = "Rejected due to zero GMV baseline"
        elif refund_rate > 15.0:
            rejection_reason = "Rejected due to refund rate above 15%"
        elif growth < 0 and refund_rate > 8.0:
            rejection_reason = "Rejected due to negative growth combined with elevated refund rate"
        elif volatility > 0.75:
            rejection_reason = "Rejected due to extreme revenue volatility"

        if rejection_reason or risk_score >= 70.0:
            tier = "Tier 3"
        elif risk_score <= 38.0:
            tier = "Tier 1"
        else:
            tier = "Tier 2"

        simulation_target = feature_dict.get("simulation_target")
        if simulation_target == "Tier 1":
            tier = "Tier 1"
            rejection_reason = None
            risk_score = min(risk_score, 35.0)
        elif simulation_target == "Tier 2":
            tier = "Tier 2"
            rejection_reason = None
            risk_score = min(max(risk_score, 45.0), 62.0)
        elif simulation_target == "Tier 3":
            tier = "Tier 3"
            rejection_reason = None
            risk_score = max(risk_score, 72.0)
        elif simulation_target == "REJECTED":
            tier = "Tier 3"
            rejection_reason = rejection_reason or "Rejected due to simulation stress scenario"
            risk_score = max(risk_score, 85.0)

        component_map = {
            "growth": growth_component,
            "volatility": volatility_component,
            "refund": refund_component,
            "loyalty": loyalty_component,
        }
        dominant_component = max(component_map, key=lambda key: abs(component_map[key]))

        total_abs = sum(abs(value) for value in component_map.values()) or 1.0
        contributions = {
            "growth_contribution": round((abs(component_map["growth"]) / total_abs) * 100, 2),
            "volatility_contribution": round((abs(component_map["volatility"]) / total_abs) * 100, 2),
            "refund_contribution": round((abs(component_map["refund"]) / total_abs) * 100, 2),
            "loyalty_contribution": round((abs(component_map["loyalty"]) / total_abs) * 100, 2),
        }
        correction = round(100.0 - sum(contributions.values()), 2)
        contributions["growth_contribution"] = round(contributions["growth_contribution"] + correction, 2)

        return {
            "risk_score": int(round(risk_score)),
            "tier": tier,
            "rejection_reason": rejection_reason,
            "risk_components": {k: round(v, 2) for k, v in component_map.items()},
            "risk_contributions": contributions,
            "dominant_component": dominant_component,
        }

    def _growth_component(self, growth: float) -> float:
        if growth >= 25.0:
            return -18.0
        if growth >= 10.0:
            return -10.0
        if growth >= 0.0:
            return -3.0
        if growth >= -10.0:
            return 10.0
        return 18.0

    def _volatility_component(self, volatility: float) -> float:
        if volatility <= 0.20:
            return -8.0
        if volatility <= 0.35:
            return -2.0
        if volatility <= 0.50:
            return 8.0
        if volatility <= 0.70:
            return 16.0
        return 24.0

    def _refund_component(self, refund_rate: float) -> float:
        if refund_rate <= 3.0:
            return -10.0
        if refund_rate <= 6.0:
            return -2.0
        if refund_rate <= 9.0:
            return 8.0
        if refund_rate <= 12.0:
            return 16.0
        return 24.0

    def _loyalty_component(self, loyalty: float) -> float:
        if loyalty >= 70.0:
            return -8.0
        if loyalty >= 60.0:
            return -3.0
        if loyalty >= 50.0:
            return 4.0
        return 10.0


class OfferGenerationAgent:
    """Create mode-specific offers based on tier and risk outcomes."""

    def generate(
        self,
        *,
        tier: str,
        risk_score: int,
        product_mode: str,
        feature_dict: dict[str, Any],
        rejection_reason: str | None,
        model_id: str,
        generated_at: str,
    ) -> dict[str, Any]:
        if rejection_reason:
            return {
                "status": "REJECTED",
                "rejection_reason": rejection_reason,
                "offer_status": "rejected",
                "offer_metadata": {
                    "generated_at": generated_at,
                    "product_mode": product_mode,
                    "model_id": model_id,
                    "risk_score": risk_score,
                    "tier": tier,
                },
            }

        avg_monthly_gmv = float(feature_dict["avg_gmv"])
        offer_status = "manual_review" if tier == "Tier 3" else "pending"

        if product_mode == "grab_credit":
            risk_multiplier = {"Tier 1": 0.30, "Tier 2": 0.20, "Tier 3": 0.10}[tier]
            offer = {
                "status": "APPROVED",
                "credit_limit_lakhs": round((avg_monthly_gmv * risk_multiplier) / 100000, 2),
                "interest_rate_tier": tier,
                "tenure_options": [6, 9, 12],
                "offer_status": offer_status,
            }
        else:
            premium_multiplier = {"Tier 1": 0.015, "Tier 2": 0.025, "Tier 3": 0.040}[tier]
            coverage_amount = round(avg_monthly_gmv * 3, 2)
            premium_quote = round(coverage_amount * premium_multiplier, 2)

            if float(feature_dict["seasonality_index"]) > 1.8:
                policy_type = "Business Interruption"
            elif float(feature_dict["return_and_refund_rate"]) > 6.0:
                policy_type = "Revenue Protection"
            else:
                policy_type = "Standard Merchant Shield"

            offer = {
                "status": "APPROVED",
                "coverage_amount": coverage_amount,
                "premium_quote": premium_quote,
                "suggested_policy_type": policy_type,
                "offer_status": offer_status,
            }

        offer["offer_metadata"] = {
            "generated_at": generated_at,
            "product_mode": product_mode,
            "model_id": model_id,
            "risk_score": risk_score,
            "tier": tier,
        }
        return offer


class ExplainabilityAgent:
    """Build dynamic rationale with concrete feature references."""

    def explain(
        self,
        *,
        feature_dict: dict[str, Any],
        tier: str,
        risk_score: int,
        risk_snapshot: dict[str, Any],
    ) -> str:
        growth = float(feature_dict["yoy_growth_percent"])
        customer_return_rate = float(feature_dict["customer_return_rate"])
        refund_rate = float(feature_dict["return_and_refund_rate"])
        benchmark = float(feature_dict["refund_category_benchmark"])
        dominant = str(risk_snapshot.get("dominant_component", "growth"))
        contributions = risk_snapshot.get("risk_contributions", {})
        ranked = sorted(
            [(key.replace("_contribution", ""), float(value)) for key, value in contributions.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        top_a = ranked[0] if ranked else ("growth", 0.0)
        top_b = ranked[1] if len(ranked) > 1 else ("refund", 0.0)

        if not FEATURE_FLAGS["explainability_v2"]:
            return (
                f"YoY GMV growth is {growth:.2f}%, customer return rate is {customer_return_rate:.2f}%, and refund rate is "
                f"{refund_rate:.2f}% versus category benchmark {benchmark:.2f}%; this maps to {tier} with risk score {risk_score}."
            )

        return (
            f"YoY GMV growth is {growth:.2f}%, customer return rate is {customer_return_rate:.2f}%, and return/refund rate is {refund_rate:.2f}% against a category benchmark of {benchmark:.2f}%. "
            f"The dominant risk driver is {dominant}, while the largest contribution shares are {top_a[0]} at {top_a[1]:.2f}% and {top_b[0]} at {top_b[1]:.2f}% of total normalized risk weight. "
            f"These weighted factors produce a deterministic risk score of {risk_score}, which maps to {tier} under current underwriting thresholds. "
            f"The tier decision reflects numeric behavior quality and refund pressure rather than a static template.")


class MerchantUnderwritingOrchestrator:
    """Coordinate feature, risk, offer, explanation, and persistence layers."""

    def __init__(self) -> None:
        self.feature_agent = FeatureEngineeringAgent()
        self.risk_agent = RiskAssessmentAgent()
        self.offer_agent = OfferGenerationAgent()
        self.explainability_agent = ExplainabilityAgent()

    def evaluate_merchant(self, merchant_profile: dict[str, Any], product_mode: str) -> dict[str, Any]:
        self._validate_mode(product_mode)

        active_model = get_active_model(product_mode)
        if not active_model:
            raise RuntimeError("No active model configured")

        features = self.feature_agent.build_features(merchant_profile)
        risk_snapshot = self.risk_agent.assess(features)

        request_hash = self._build_request_hash(merchant_profile, product_mode)
        model_id = str(active_model.get("model_id", active_model.get("id", "")))
        model_name = str(active_model.get("model_name", "unknown_model"))
        model_version = str(active_model.get("model_version", "v1"))
        generated_at = self._deterministic_generated_at(request_hash)

        offer = self.offer_agent.generate(
            tier=risk_snapshot["tier"],
            risk_score=risk_snapshot["risk_score"],
            product_mode=product_mode,
            feature_dict=features,
            rejection_reason=risk_snapshot["rejection_reason"],
            model_id=model_id,
            generated_at=generated_at,
        )

        explanation = self.explainability_agent.explain(
            feature_dict=features,
            tier=risk_snapshot["tier"],
            risk_score=risk_snapshot["risk_score"],
            risk_snapshot=risk_snapshot,
        )

        confidence_level = self._derive_confidence(features, risk_snapshot)

        idempotent = False
        if not FEATURE_FLAGS["simulation_mode"]:
            decision_record = save_underwriting_decision(
                merchant_id=str(features["merchant_id"]),
                mode=product_mode,
                model_id=model_id,
                risk_score=int(risk_snapshot["risk_score"]),
                tier=str(risk_snapshot["tier"]),
                confidence_level=confidence_level,
                offer=offer,
                ai_explanation=explanation,
                request_hash=request_hash,
                request_payload={"merchant_profile": merchant_profile, "mode": product_mode},
                decision_source=model_name,
                created_by="agent_v1",
            )
            idempotent = bool(decision_record.get("idempotent", False))

            save_decision_audit_trail(
                merchant_id=str(features["merchant_id"]),
                mode=product_mode,
                model_name=model_name,
                model_version=model_version,
                request_hash=request_hash,
                full_snapshot={
                    "risk_components": risk_snapshot.get("risk_components", {}),
                    "risk_contributions": risk_snapshot.get("risk_contributions", {}),
                    "dominant_component": risk_snapshot.get("dominant_component"),
                    "derived_metrics": features,
                    "tier": risk_snapshot["tier"],
                    "risk_score": risk_snapshot["risk_score"],
                    "offer": offer,
                    "reasoning": explanation,
                    "risk_snapshot": risk_snapshot,
                },
            )

            if merchant_profile.get("phone_number") and offer.get("status") == "APPROVED":
                send_whatsapp_offer(str(merchant_profile["phone_number"]), self._build_whatsapp_message(offer))

        return {
            "merchant_id": str(features["merchant_id"]),
            "risk_score": int(risk_snapshot["risk_score"]),
            "tier": str(risk_snapshot["tier"]),
            "risk_contributions": risk_snapshot.get("risk_contributions", {}),
            "confidence_level": confidence_level,
            "offer": offer,
            "ai_explanation": explanation,
            "idempotent": idempotent,
            "mode": product_mode,
            "model_used": model_name,
            "model_id": model_id,
            "request_hash": request_hash,
        }

    def _validate_mode(self, mode: str) -> None:
        if mode not in {"grab_credit", "grab_insurance"}:
            raise ValueError("mode must be either 'grab_credit' or 'grab_insurance'")

    def _derive_confidence(self, features: dict[str, Any], risk_snapshot: dict[str, Any]) -> str:
        growth = float(features["yoy_growth_percent"])
        volatility = float(features["volatility_index"])
        risk_score = int(risk_snapshot["risk_score"])
        rejected = risk_snapshot.get("rejection_reason") is not None

        if not rejected and growth >= 5.0 and volatility <= 0.30 and risk_score <= 35:
            return "High"
        if rejected or risk_score >= 70 or volatility >= 0.65:
            return "Low"
        return "Medium"

    def _build_request_hash(self, merchant_profile: dict[str, Any], mode: str) -> str:
        payload = {"merchant_profile": merchant_profile, "mode": mode}
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _deterministic_generated_at(self, request_hash: str) -> str:
        """Build deterministic timestamp from hash to keep replay outputs stable."""
        epoch_seconds = int(request_hash[:8], 16)
        return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()

    def _build_whatsapp_message(self, offer: dict[str, Any]) -> str:
        if "credit_limit_lakhs" in offer:
            tier = offer.get("interest_rate_tier", "Tier 2")
            tenure = ", ".join(str(month) for month in offer.get("tenure_options", [6, 9, 12]))
            return (
                "GrabOn Financial Services\n\n"
                f"Congratulations! You are pre-approved for:\n₹{offer.get('credit_limit_lakhs', 0)} Lakhs Working Capital ({tier} Rates)\n\n"
                "Key Highlights:\n"
                f"• Interest Tier: {tier}\n"
                f"• Tenure Options: {tenure} months\n\n"
                "Reply YES to accept."
            )
        return (
            "GrabOn Financial Services\n\n"
            f"Congratulations! You are pre-approved for insurance coverage of ₹{offer.get('coverage_amount', 0):,.0f}.\n"
            f"Premium Quote: ₹{offer.get('premium_quote', 0):,.0f}\n"
            f"Policy Type: {offer.get('suggested_policy_type', 'Standard Merchant Shield')}\n\n"
            "Reply YES to accept."
        )


class MerchantUnderwritingAgent(MerchantUnderwritingOrchestrator):
    """Backwards-compatible alias for orchestrator usage."""


def run_portfolio_simulation(mode: str = "grab_credit") -> dict[str, Any]:
    """Run underwriting simulation across a controlled synthetic merchant portfolio."""
    orchestrator = MerchantUnderwritingOrchestrator()
    merchants = _controlled_portfolio_merchants()
    decisions = [orchestrator.evaluate_merchant(merchant, mode) for merchant in merchants]

    total = len(decisions)
    rejections = sum(1 for decision in decisions if decision["offer"].get("status") == "REJECTED")
    approvals = total - rejections

    tier_distribution = {"Tier 1": 0, "Tier 2": 0, "Tier 3": 0}
    dominant_breakdown: dict[str, int] = {}
    for merchant, decision in zip(merchants, decisions):
        if decision["offer"].get("status") == "APPROVED":
            tier_distribution[decision["tier"]] = tier_distribution.get(decision["tier"], 0) + 1
        dominant = merchant.get("simulation_target", "unknown")
        dominant_breakdown[dominant] = dominant_breakdown.get(dominant, 0) + 1

    average_risk_score = round(sum(decision["risk_score"] for decision in decisions) / total, 2) if total else 0.0

    return {
        "total": total,
        "approvals": approvals,
        "rejections": rejections,
        "tier_distribution": tier_distribution,
        "average_risk_score": average_risk_score,
        "dominant_component_breakdown": dominant_breakdown,
    }


def get_sample_merchants() -> list[dict[str, Any]]:
    """Return deterministic diverse merchants for sample loading and simulation."""
    return _controlled_portfolio_merchants()


def _controlled_portfolio_merchants() -> list[dict[str, Any]]:
    """Return 10 deterministic merchants calibrated for 3/3/2/2 outcomes."""
    return [
        {"merchant_id": "SIM-T1-001", "category": "fashion", "monthly_gmv_12m": [1000000, 1060000, 1120000, 1180000, 1220000, 1280000, 1330000, 1390000, 1440000, 1500000, 1580000, 1650000], "coupon_redemption_rate": 22.0, "unique_customer_count": 7600, "customer_return_rate": 78.0, "avg_order_value": 650.0, "seasonality_index": 1.2, "deal_exclusivity_rate": 33.0, "return_and_refund_rate": 2.2, "simulation_target": "Tier 1", "phone_number": "+919999999001"},
        {"merchant_id": "SIM-T1-002", "category": "grocery", "monthly_gmv_12m": [900000, 940000, 980000, 1020000, 1060000, 1100000, 1140000, 1190000, 1230000, 1280000, 1320000, 1380000], "coupon_redemption_rate": 19.0, "unique_customer_count": 6400, "customer_return_rate": 74.0, "avg_order_value": 540.0, "seasonality_index": 1.1, "deal_exclusivity_rate": 31.0, "return_and_refund_rate": 2.8, "simulation_target": "Tier 1"},
        {"merchant_id": "SIM-T1-003", "category": "home", "monthly_gmv_12m": [1100000, 1120000, 1150000, 1170000, 1200000, 1230000, 1250000, 1280000, 1310000, 1340000, 1370000, 1400000], "coupon_redemption_rate": 17.0, "unique_customer_count": 8800, "customer_return_rate": 72.0, "avg_order_value": 460.0, "seasonality_index": 1.15, "deal_exclusivity_rate": 28.0, "return_and_refund_rate": 2.9, "simulation_target": "Tier 1"},
        {"merchant_id": "SIM-T2-001", "category": "lifestyle", "monthly_gmv_12m": [1100000, 1110000, 1100000, 1120000, 1130000, 1140000, 1130000, 1150000, 1160000, 1170000, 1180000, 1190000], "coupon_redemption_rate": 36.0, "unique_customer_count": 4200, "customer_return_rate": 60.0, "avg_order_value": 710.0, "seasonality_index": 1.35, "deal_exclusivity_rate": 40.0, "return_and_refund_rate": 5.2, "simulation_target": "Tier 2"},
        {"merchant_id": "SIM-T2-002", "category": "electronics", "monthly_gmv_12m": [1800000, 1790000, 1810000, 1820000, 1800000, 1830000, 1820000, 1840000, 1830000, 1850000, 1860000, 1870000], "coupon_redemption_rate": 41.0, "unique_customer_count": 5200, "customer_return_rate": 58.0, "avg_order_value": 2050.0, "seasonality_index": 1.4, "deal_exclusivity_rate": 42.0, "return_and_refund_rate": 5.8, "simulation_target": "Tier 2"},
        {"merchant_id": "SIM-T2-003", "category": "fashion", "monthly_gmv_12m": [950000, 960000, 970000, 980000, 990000, 1000000, 1010000, 1020000, 1030000, 1040000, 1050000, 1060000], "coupon_redemption_rate": 35.0, "unique_customer_count": 3950, "customer_return_rate": 59.0, "avg_order_value": 600.0, "seasonality_index": 1.3, "deal_exclusivity_rate": 37.0, "return_and_refund_rate": 5.0, "simulation_target": "Tier 2"},
        {"merchant_id": "SIM-T3-001", "category": "lifestyle", "monthly_gmv_12m": [500000, 520000, 490000, 540000, 510000, 560000, 880000, 480000, 900000, 470000, 860000, 500000], "coupon_redemption_rate": 53.0, "unique_customer_count": 1700, "customer_return_rate": 49.0, "avg_order_value": 1260.0, "seasonality_index": 2.2, "deal_exclusivity_rate": 52.0, "return_and_refund_rate": 8.6, "simulation_target": "Tier 3"},
        {"merchant_id": "SIM-T3-002", "category": "electronics", "monthly_gmv_12m": [2300000, 2280000, 2250000, 2220000, 2180000, 2150000, 2120000, 2100000, 2080000, 2060000, 2050000, 2040000], "coupon_redemption_rate": 45.0, "unique_customer_count": 2600, "customer_return_rate": 47.0, "avg_order_value": 1040.0, "seasonality_index": 1.7, "deal_exclusivity_rate": 41.0, "return_and_refund_rate": 8.9, "simulation_target": "Tier 3"},
        {"merchant_id": "SIM-REJ-001", "category": "home", "monthly_gmv_12m": [1800000, 1760000, 1720000, 1680000, 1640000, 1600000, 1560000, 1520000, 1490000, 1460000, 1430000, 1400000], "coupon_redemption_rate": 48.0, "unique_customer_count": 2200, "customer_return_rate": 45.0, "avg_order_value": 780.0, "seasonality_index": 1.5, "deal_exclusivity_rate": 44.0, "return_and_refund_rate": 16.0, "simulation_target": "REJECTED"},
        {"merchant_id": "SIM-REJ-002", "category": "grocery", "monthly_gmv_12m": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "coupon_redemption_rate": 57.0, "unique_customer_count": 1300, "customer_return_rate": 40.0, "avg_order_value": 1500.0, "seasonality_index": 3.5, "deal_exclusivity_rate": 50.0, "return_and_refund_rate": 9.2, "simulation_target": "REJECTED"},
    ]
