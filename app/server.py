"""Lightweight FastAPI server for underwriting dashboard flows."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from app.explainability import CategoryAverages, generate_underwriting_decision_trail
from app.portfolio import generate_portfolio_summary
from app.underwriting import generate_underwriting_decision
from app.whatsapp import WhatsAppDeliveryError, send_whatsapp_offer

UnderwritingMode = Literal["grab_credit", "grab_insurance"]


@dataclass
class MerchantStub:
    """In-memory merchant model compatible with underwriting protocol."""

    merchant_id: str
    category: str
    monthly_gmv_12m: list[float]
    coupon_redemption_rate: float
    unique_customer_count: int
    customer_return_rate: float
    avg_order_value: float
    seasonality_index: float
    deal_exclusivity_rate: float
    return_and_refund_rate: float
    phone_number: str | None = None

    def compute_yoy_gmv_growth(self) -> float:
        start = self.monthly_gmv_12m[0]
        end = self.monthly_gmv_12m[-1]
        if start == 0:
            raise ValueError("Cannot compute YoY growth when baseline GMV is zero")
        return ((end - start) / start) * 100

    def compute_gmv_volatility(self) -> float:
        return stdev(self.monthly_gmv_12m)


class MerchantInput(BaseModel):
    """Validated merchant input schema for on-demand underwriting requests."""

    merchant_id: str
    category: str
    monthly_gmv_12m: list[float] = Field(..., min_length=12, max_length=12)
    coupon_redemption_rate: float
    unique_customer_count: int
    customer_return_rate: float
    avg_order_value: float
    seasonality_index: float
    deal_exclusivity_rate: float
    return_and_refund_rate: float
    phone_number: str | None = None

    @field_validator("monthly_gmv_12m")
    @classmethod
    def validate_monthly_gmv_values(cls, values: list[float]) -> list[float]:
        """Ensure exactly 12 non-negative monthly GMV values are provided."""
        if len(values) != 12:
            raise ValueError("monthly_gmv_12m must contain exactly 12 months")
        if any(value < 0 for value in values):
            raise ValueError("monthly_gmv_12m values must be non-negative")
        return values


BASELINE_CATEGORY_AVERAGES = CategoryAverages(
    refund_rate=4.5,
    customer_return_rate=60.0,
    yoy_gmv_growth=10.0,
)


def build_demo_merchants() -> list[MerchantStub]:
    """Return exactly ten diverse demo merchants for underwriting scenarios."""
    return [
        MerchantStub("M-FASHION-T1", "fashion", [1_000_000, 1_070_000, 1_110_000, 1_180_000, 1_240_000, 1_300_000, 1_360_000, 1_430_000, 1_500_000, 1_620_000, 1_760_000, 1_980_000], 22.0, 8_200, 74.0, 650.0, 1.25, 33.0, 2.2, "+919100000001"),
        MerchantStub("M-PHARMA-T1", "pharma", [1_300_000, 1_360_000, 1_420_000, 1_500_000, 1_560_000, 1_620_000, 1_680_000, 1_760_000, 1_850_000, 1_960_000, 2_080_000, 2_220_000], 20.0, 9_100, 79.0, 560.0, 1.15, 28.0, 1.9, "+919100000002"),
        MerchantStub("M-GROCERY-T2", "grocery", [1_600_000, 1_620_000, 1_610_000, 1_640_000, 1_660_000, 1_650_000, 1_670_000, 1_690_000, 1_700_000, 1_710_000, 1_720_000, 1_730_000], 36.0, 4_200, 62.0, 500.0, 1.20, 40.0, 5.1, "+919100000003"),
        MerchantStub("M-BEAUTY-T2", "beauty", [1_100_000, 1_120_000, 1_130_000, 1_140_000, 1_160_000, 1_170_000, 1_190_000, 1_200_000, 1_210_000, 1_220_000, 1_240_000, 1_260_000], 34.0, 3_700, 60.0, 470.0, 1.45, 36.0, 4.2, "+919100000004"),
        MerchantStub("M-EDU-T2", "education", [900_000, 910_000, 930_000, 940_000, 960_000, 980_000, 990_000, 1_020_000, 1_030_000, 1_050_000, 1_070_000, 1_090_000], 31.0, 3_000, 61.0, 420.0, 2.30, 34.0, 4.0, "+919100000005"),
        MerchantStub("M-ELEC-T3", "electronics", [15_000_000, 14_800_000, 14_900_000, 14_700_000, 14_850_000, 14_650_000, 14_900_000, 14_600_000, 14_820_000, 14_640_000, 14_780_000, 14_600_000], 52.0, 30_000, 51.0, 2_100.0, 1.70, 45.0, 6.2, "+919100000006"),
        MerchantStub("M-FOOD-T3", "food", [1_500_000, 1_350_000, 1_300_000, 1_400_000, 1_480_000, 1_520_000, 1_800_000, 1_950_000, 1_900_000, 1_420_000, 1_360_000, 1_330_000], 47.0, 5_800, 56.0, 780.0, 2.60, 41.0, 5.9, "+919100000007"),
        MerchantStub("M-TRAVEL-T3", "travel", [2_500_000, 2_420_000, 2_360_000, 2_300_000, 2_240_000, 2_180_000, 2_140_000, 2_100_000, 2_060_000, 2_020_000, 1_980_000, 1_940_000], 40.0, 3_300, 49.0, 980.0, 2.20, 38.0, 4.6, "+919100000008"),
        MerchantStub("M-HOME-REJ", "home_decor", [2_200_000, 2_170_000, 2_150_000, 2_120_000, 2_100_000, 2_080_000, 2_050_000, 2_030_000, 2_010_000, 1_990_000, 1_970_000, 1_950_000], 49.0, 2_500, 50.0, 690.0, 1.90, 42.0, 8.8, "+919100000009"),
        MerchantStub("M-EVENT-REJ", "events", [450_000, 480_000, 470_000, 500_000, 520_000, 490_000, 2_000_000, 2_200_000, 2_050_000, 530_000, 500_000, 470_000], 58.0, 1_700, 54.0, 1_250.0, 4.10, 48.0, 4.3, "+919100000010"),
    ]


def _build_decision_payload(merchant: MerchantStub, mode: UnderwritingMode) -> dict:
    """Build decision payload with explanation and operational flags."""
    decision = generate_underwriting_decision(
        merchant=merchant,
        mode=mode,
        category_averages=BASELINE_CATEGORY_AVERAGES,
    )
    trail = generate_underwriting_decision_trail(
        merchant=merchant,
        category_averages=BASELINE_CATEGORY_AVERAGES,
        mode=mode,
    )
    return {
        **decision,
        "ai_explanation": trail.get("final_explanation", ""),
        "primary_risk_drivers": trail.get("primary_risk_drivers", []),
        "primary_strengths": trail.get("primary_strengths", []),
        "confidence_score": trail.get("confidence_score", decision.get("risk_score", 0.0)),
        "offer_sent": False,
    }


def _load_portfolio_decisions(mode: UnderwritingMode) -> list[dict]:
    """Generate decisions for all demo merchants under a selected mode."""
    return [_build_decision_payload(merchant, mode=mode) for merchant in merchants]


def _find_merchant(merchant_id: str) -> MerchantStub | None:
    for merchant in merchants:
        if merchant.merchant_id == merchant_id:
            return merchant
    return None


def _find_decision(merchant_id: str) -> dict | None:
    for decision in portfolio_decisions:
        if decision.get("merchant_id") == merchant_id:
            return decision
    return None


def build_offer_message(decision: dict, ai_explanation: str) -> str:
    """Build mode-aware WhatsApp message content from decision payload."""
    offer = decision.get("offer", {})
    mode = offer.get("mode", "grab_credit")
    tier = offer.get("tier", "N/A")
    confidence_level = decision.get("confidence_level", "N/A")

    if mode == "grab_insurance":
        mode_block = (
            "GrabInsurance Offer Ready\n"
            f"Coverage: ₹{offer.get('coverage_amount_lakhs')} L\n"
            f"Premium: {offer.get('premium_quote_pct')}% p.a.\n"
            f"Policy: {offer.get('policy_type')}"
        )
    else:
        mode_block = (
            "GrabCredit Offer Approved\n"
            f"Tier: {tier}\n"
            f"Limit: ₹{offer.get('credit_limit_lakhs')} L\n"
            f"Rate: {offer.get('interest_rate_pct')}%"
        )

    return (
        f"{mode_block}\n"
        f"Confidence: {confidence_level}\n\n"
        f"Explanation: {ai_explanation}\n\n"
        "Reply YES to accept or visit dashboard"
    )


app = FastAPI(title="Merchant Underwriting Dashboard API")

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))
merchants = build_demo_merchants()
portfolio_decisions: list[dict] = []
active_mode: UnderwritingMode = "grab_credit"
accepted_merchants: set[str] = set()


@app.on_event("startup")
def startup_load_decisions() -> None:
    """Generate and cache credit decisions at service startup."""
    global portfolio_decisions, active_mode
    active_mode = "grab_credit"
    portfolio_decisions = _load_portfolio_decisions(active_mode)


def _dashboard_merchant_item(decision: dict, mode: UnderwritingMode) -> dict:
    offer = decision.get("offer", {})
    rejected = isinstance(offer, dict) and offer.get("status") == "REJECTED"
    merchant_id = str(decision.get("merchant_id"))
    merchant = _find_merchant(merchant_id)
    tier = offer.get("tier") if isinstance(offer, dict) else None

    return {
        "merchant_id": merchant_id,
        "category": decision.get("offer", {}).get("decision_metrics", {}).get("category"),
        "phone_number": merchant.phone_number if merchant else None,
        "mode": str(offer.get("mode", mode)),
        "tier": tier,
        "confidence_score": round(float(decision.get("confidence_score", 0.0)), 2),
        "confidence_level": decision.get("confidence_level"),
        "status": "rejected" if rejected else "approved",
        "credit_limit_lakhs": None if rejected else offer.get("credit_limit_lakhs"),
        "interest_rate_pct": None if rejected else offer.get("interest_rate_pct"),
        "coverage_amount_lakhs": None if rejected else offer.get("coverage_amount_lakhs"),
        "premium_quote_pct": None if rejected else offer.get("premium_quote_pct"),
        "policy_type": None if rejected else offer.get("policy_type"),
        "rejection_reason": offer.get("rejection_reason") if rejected else None,
        "accepted": merchant_id in accepted_merchants,
        "offer_sent": bool(decision.get("offer_sent", False)),
        "primary_risk_drivers": decision.get("primary_risk_drivers", []),
        "primary_strengths": decision.get("primary_strengths", []),
        "ai_explanation_full": decision.get("ai_explanation", ""),
        "decision_metrics": offer.get("decision_metrics", {}),
        "underwriting_mode": mode,
        "tier_class": (
            "tier-1" if tier == "Tier 1" else "tier-2" if tier == "Tier 2" else "tier-3" if tier == "Tier 3" else "rejected"
        ),
    }


@app.get("/dashboard")
def get_dashboard(mode: UnderwritingMode = Query("grab_credit")) -> dict:
    """Return portfolio summary and merchant-level dashboard rows."""
    decisions = portfolio_decisions if mode == active_mode else _load_portfolio_decisions(mode)
    return {
        "mode": mode,
        "portfolio_summary": generate_portfolio_summary(decisions),
        "merchants": [_dashboard_merchant_item(decision, mode=mode) for decision in decisions],
    }


@app.get("/ui", response_class=HTMLResponse)
def dashboard_ui(request: Request, mode: UnderwritingMode = Query("grab_credit")) -> HTMLResponse:
    """Render simple HTML dashboard."""
    decisions = portfolio_decisions if mode == active_mode else _load_portfolio_decisions(mode)
    context = {
        "request": request,
        "mode": mode,
        "portfolio_summary": generate_portfolio_summary(decisions),
        "merchants": [_dashboard_merchant_item(decision, mode=mode) for decision in decisions],
    }
    return templates.TemplateResponse("dashboard.html", context)


@app.post("/send-offer/{merchant_id}")
def send_offer(merchant_id: str) -> dict:
    """Send approved offer via WhatsApp using mode-aware message formatting."""
    decision = _find_decision(merchant_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Merchant not found")

    if merchant_id in accepted_merchants:
        raise HTTPException(status_code=409, detail="Already accepted")

    offer = decision.get("offer", {})
    if isinstance(offer, dict) and offer.get("status") == "REJECTED":
        raise HTTPException(status_code=400, detail="Cannot send offer to rejected merchant")

    merchant = _find_merchant(merchant_id)
    phone_number = merchant.phone_number if merchant else None
    if not phone_number:
        return {
            "status": "failed",
            "message_sid": None,
            "delivery_info": {"reason": "Missing phone_number; skipped send"},
        }

    ai_explanation = str(decision.get("ai_explanation", "Not available"))
    message_content = build_offer_message(decision=decision, ai_explanation=ai_explanation)

    try:
        delivery_info = send_whatsapp_offer(
            merchant_id=merchant_id,
            phone_number=phone_number,
            message=message_content,
        )
    except WhatsAppDeliveryError as exc:
        return {
            "status": "failed",
            "message_sid": None,
            "delivery_info": {"reason": str(exc)},
        }

    decision["offer_sent"] = True
    return {
        "status": "sent",
        "message_sid": delivery_info.get("message_id") or delivery_info.get("sid"),
        "delivery_info": delivery_info,
    }


@app.post("/accept-offer/{merchant_id}")
def accept_offer(merchant_id: str) -> dict:
    """Mark merchant offer as accepted and return mock NACH initiation payload."""
    decision = _find_decision(merchant_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Merchant not found")

    offer = decision.get("offer", {})
    decision_status = "rejected" if isinstance(offer, dict) and offer.get("status") == "REJECTED" else "approved"
    if decision_status != "approved":
        raise HTTPException(
            status_code=400,
            detail="Offer cannot be accepted because merchant was not approved",
        )

    if not bool(decision.get("offer_sent", False)):
        raise HTTPException(
            status_code=409,
            detail="Offer must be sent on WhatsApp before acceptance",
        )

    accepted_merchants.add(merchant_id)
    return {
        "merchant_id": merchant_id,
        "status": "mandate_initiated",
        "mandate_reference": f"MOCK-NACH-{random.randint(1000, 9999)}",
    }


@app.post("/underwrite")
def underwrite_merchant(payload: MerchantInput, mode: UnderwritingMode = Query("grab_credit")) -> dict:
    """Run underwriting for an input merchant and return decision + AI explanation."""
    try:
        merchant = MerchantStub(**payload.model_dump())
        decision = generate_underwriting_decision(
            merchant=merchant,
            mode=mode,
            category_averages=BASELINE_CATEGORY_AVERAGES,
        )
        trail = generate_underwriting_decision_trail(
            merchant=merchant,
            category_averages=BASELINE_CATEGORY_AVERAGES,
            mode=mode,
        )

        offer = decision.get("offer", {})
        tier = offer.get("tier") if isinstance(offer, dict) else None

        return {
            **decision,
            "tier": tier,
            "mode": mode,
            "ai_explanation": trail.get("final_explanation"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to underwrite merchant: {exc}",
        ) from exc
