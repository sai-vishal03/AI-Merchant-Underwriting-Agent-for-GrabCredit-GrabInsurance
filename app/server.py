"""Lightweight FastAPI server for underwriting dashboard flows."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from app.demo_data import MerchantStub, build_demo_merchants, build_offer_message
from app.explainability import CategoryAverages, generate_underwriting_decision_trail
from app.portfolio import generate_portfolio_summary
from app.underwriting import generate_underwriting_decision
from app.whatsapp import WhatsAppDeliveryError, send_whatsapp_offer

UnderwritingMode = Literal["grab_credit", "grab_insurance"]


class MerchantInput(BaseModel):
    """Validated merchant payload schema for on-demand underwriting requests."""

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

    @field_validator("monthly_gmv_12m")
    @classmethod
    def validate_monthly_gmv_values(cls, values: list[float]) -> list[float]:
        """Ensure exactly 12 non-negative monthly GMV values are provided."""
        if len(values) != 12:
            raise ValueError("monthly_gmv_12m must contain exactly 12 values")
        if any(value < 0 for value in values):
            raise ValueError("monthly_gmv_12m values must be non-negative")
        return values

    @field_validator(
        "coupon_redemption_rate",
        "unique_customer_count",
        "customer_return_rate",
        "avg_order_value",
        "seasonality_index",
        "deal_exclusivity_rate",
        "return_and_refund_rate",
    )
    @classmethod
    def validate_non_negative_numeric_fields(cls, value: float | int) -> float | int:
        """Enforce non-negative numeric payload fields."""
        if value < 0:
            raise ValueError("numeric fields must be non-negative")
        return value


BASELINE_CATEGORY_AVERAGES = CategoryAverages(
    refund_rate=4.5,
    customer_return_rate=60.0,
    yoy_gmv_growth=10.0,
)

app = FastAPI(title="Merchant Underwriting Dashboard API")

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))
merchants = build_demo_merchants()
portfolio_decisions: list[dict] = []
cached_mode: UnderwritingMode = "grab_credit"
accepted_merchants: set[str] = set()


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
    return next((merchant for merchant in merchants if merchant.merchant_id == merchant_id), None)


def _find_decision(merchant_id: str, decisions: list[dict]) -> dict | None:
    return next((decision for decision in decisions if decision.get("merchant_id") == merchant_id), None)


def _decisions_for_mode(mode: UnderwritingMode) -> list[dict]:
    """Return cached decisions, regenerating when requested mode changes."""
    global cached_mode
    if not portfolio_decisions or cached_mode != mode:
        portfolio_decisions.clear()
        portfolio_decisions.extend(_load_portfolio_decisions(mode))
        cached_mode = mode
    return portfolio_decisions


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
        "tier_class": (
            "tier-1" if tier == "Tier 1" else "tier-2" if tier == "Tier 2" else "tier-3" if tier == "Tier 3" else "rejected"
        ),
    }


def initialize_caches() -> None:
    """Generate and cache default portfolio decisions for dashboard actions."""
    global cached_mode
    portfolio_decisions.clear()
    portfolio_decisions.extend(_load_portfolio_decisions("grab_credit"))
    cached_mode = "grab_credit"


@app.on_event("startup")
def startup_load_decisions() -> None:
    """Initialize in-memory cache at service startup."""
    initialize_caches()


@app.get("/health")
def health() -> dict:
    """Simple health endpoint for deployment and smoke checks."""
    return {"status": "ok", "merchants": len(merchants)}


@app.post("/refresh-decisions")
def refresh_decisions(mode: UnderwritingMode | None = Query(None)) -> dict:
    """Refresh cached decisions (credit cache only) and report refresh metadata."""
    target_mode = mode or "grab_credit"
    portfolio_decisions.clear()
    portfolio_decisions.extend(_load_portfolio_decisions(target_mode))
    global cached_mode
    cached_mode = target_mode
    return {
        "status": "refreshed",
        "mode": target_mode,
        "cached_merchants": len(portfolio_decisions),
    }


@app.get("/dashboard")
def get_dashboard(mode: UnderwritingMode = Query("grab_credit")) -> dict:
    """Return portfolio summary and merchant-level dashboard rows."""
    decisions = _decisions_for_mode(mode)
    return {
        "mode": mode,
        "portfolio_summary": generate_portfolio_summary(decisions),
        "merchants": [_dashboard_merchant_item(decision, mode=mode) for decision in decisions],
    }


@app.get("/decision-trail/{merchant_id}")
def get_decision_trail(merchant_id: str, mode: UnderwritingMode = Query("grab_credit")) -> dict:
    """Return full decision and explainability trail for one merchant."""
    decisions = _decisions_for_mode(mode)
    decision = _find_decision(merchant_id, decisions)
    if decision is None:
        raise HTTPException(status_code=404, detail="Merchant not found")
    return {
        "mode": mode,
        "merchant_id": merchant_id,
        "offer_sent": bool(decision.get("offer_sent", False)),
        "accepted": merchant_id in accepted_merchants,
        "confidence_level": decision.get("confidence_level"),
        "confidence_score": decision.get("confidence_score"),
        "primary_risk_drivers": decision.get("primary_risk_drivers", []),
        "primary_strengths": decision.get("primary_strengths", []),
        "ai_explanation": decision.get("ai_explanation", ""),
        "offer": decision.get("offer", {}),
    }


@app.get("/ui", response_class=HTMLResponse)
def dashboard_ui(request: Request, mode: UnderwritingMode = Query("grab_credit")) -> HTMLResponse:
    """Render simple HTML dashboard."""
    decisions = _decisions_for_mode(mode)
    context = {
        "request": request,
        "mode": mode,
        "portfolio_summary": generate_portfolio_summary(decisions),
        "merchants": [_dashboard_merchant_item(decision, mode=mode) for decision in decisions],
    }
    return templates.TemplateResponse("dashboard.html", context)


@app.post("/send-offer/{merchant_id}")
def send_offer(merchant_id: str, mode: UnderwritingMode = Query("grab_credit")) -> dict:
    """Send approved offer via WhatsApp using mode-aware message formatting."""
    decisions = _decisions_for_mode(mode)
    decision = _find_decision(merchant_id, decisions)
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
def accept_offer(merchant_id: str, mode: UnderwritingMode = Query("grab_credit")) -> dict:
    """Mark merchant offer as accepted and return mock NACH initiation payload."""
    decisions = _decisions_for_mode(mode)
    decision = _find_decision(merchant_id, decisions)
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
def underwrite_merchant(payload: MerchantInput) -> dict:
    """Run grab_credit underwriting for an input merchant and return decision + explanation.

    This endpoint validates merchant schema with Pydantic, applies additional
    business-logic safeguards, runs underwriting with baseline category averages,
    and returns the full decision payload including risk score, tier, offer,
    confidence level, and AI explanation.
    """
    try:
        if not payload.merchant_id.strip() or not payload.category.strip():
            raise HTTPException(status_code=400, detail="merchant_id and category must be non-empty")

        merchant = MerchantStub(**payload.model_dump(), phone_number=None)

        decision = generate_underwriting_decision(
            merchant=merchant,
            mode="grab_credit",
            category_averages=BASELINE_CATEGORY_AVERAGES,
        )
        trail = generate_underwriting_decision_trail(
            merchant=merchant,
            category_averages=BASELINE_CATEGORY_AVERAGES,
            mode="grab_credit",
        )

        offer = decision.get("offer", {})
        tier = offer.get("tier") if isinstance(offer, dict) else None

        return {
            "risk_score": decision.get("risk_score"),
            "tier": tier,
            "offer": offer,
            "ai_explanation": trail.get("final_explanation"),
            "confidence_level": decision.get("confidence_level"),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to underwrite merchant: {exc}") from exc
