"""Lightweight FastAPI server for underwriting dashboard flows."""

from __future__ import annotations

import random
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from app.database import get_accepted_merchant_ids, get_decision_history, get_portfolio_analytics, get_portfolio_analytics_v2, get_portfolio_risk_alerts, initialize_database, save_accepted_offer, save_underwriting_decision, upsert_merchant
from app.explainability import CategoryAverages, generate_underwriting_decision_trail
from app.portfolio import generate_portfolio_summary
from app.underwriting import generate_underwriting_decision


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


def _build_merchants() -> list[MerchantStub]:
    """Build the same six merchants used in portfolio testing."""
    return [
        MerchantStub("M-T1-STRONG", "fashion", [1_000_000, 1_050_000, 1_120_000, 1_200_000, 1_280_000, 1_350_000, 1_420_000, 1_500_000, 1_620_000, 1_780_000, 1_920_000, 2_200_000], 22.0, 8_500, 74.0, 640.0, 1.25, 32.0, 2.1),
        MerchantStub("M-T2-HAIRCUT", "grocery", [1_800_000, 1_820_000, 1_790_000, 1_810_000, 1_850_000, 1_840_000, 1_860_000, 1_870_000, 1_880_000, 1_890_000, 1_900_000, 1_910_000], 35.0, 4_200, 63.0, 520.0, 1.20, 40.0, 5.2),
        MerchantStub("M-T3-EXPOSURE-CAP", "electronics", [80_000_000, 81_500_000, 80_800_000, 82_000_000, 81_200_000, 82_500_000, 81_900_000, 83_200_000, 82_400_000, 83_500_000, 82_700_000, 83_000_000], 58.0, 65_000, 50.0, 2_300.0, 1.45, 48.0, 7.0),
        MerchantStub("M-REJ-REFUND", "home_decor", [2_200_000, 2_150_000, 2_180_000, 2_160_000, 2_140_000, 2_130_000, 2_120_000, 2_100_000, 2_090_000, 2_080_000, 2_070_000, 2_060_000], 50.0, 2_600, 52.0, 700.0, 1.30, 44.0, 8.4),
        MerchantStub("M-REJ-NEG-GROWTH", "travel", [3_000_000, 2_850_000, 2_700_000, 2_650_000, 2_500_000, 2_400_000, 2_300_000, 2_200_000, 2_100_000, 2_000_000, 1_950_000, 1_850_000], 42.0, 3_000, 48.0, 1_050.0, 1.60, 39.0, 4.1),
        MerchantStub("M-HIGH-VOL-SEASONAL", "events", [500_000, 520_000, 510_000, 530_000, 540_000, 520_000, 1_900_000, 2_100_000, 1_850_000, 560_000, 540_000, 530_000], 60.0, 1_800, 55.0, 1_300.0, 3.80, 50.0, 3.9),
    ]


templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))
merchants = _build_merchants()
portfolio_decisions: list[dict] = []
accepted_merchants: set[str] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize database state and preload cached decisions on application startup."""
    initialize_database()
    accepted_merchants.clear()
    accepted_merchants.update(get_accepted_merchant_ids())
    portfolio_decisions.clear()
    for merchant in merchants:
        decision = generate_underwriting_decision(
            merchant=merchant,
            mode="grab_credit",
            category_averages=BASELINE_CATEGORY_AVERAGES,
        )
        portfolio_decisions.append(decision)
    yield


app = FastAPI(title="Merchant Underwriting Dashboard API", lifespan=lifespan)


def _dashboard_merchant_item(decision: dict) -> dict:
    offer = decision.get("offer", {})
    rejected = isinstance(offer, dict) and offer.get("status") == "REJECTED"
    tier = offer.get("tier") if isinstance(offer, dict) else None

    return {
        "merchant_id": decision.get("merchant_id"),
        "tier": tier,
        "confidence_level": decision.get("confidence_level"),
        "status": "rejected" if rejected else "approved",
        "credit_limit_lakhs": None if rejected else offer.get("credit_limit_lakhs"),
        "rejection_reason": offer.get("rejection_reason") if rejected else None,
        "accepted": decision.get("merchant_id") in accepted_merchants,
        "tier_class": (
            "tier-1" if tier == "Tier 1" else "tier-2" if tier == "Tier 2" else "tier-3" if tier == "Tier 3" else "rejected"
        ),
    }


@app.get("/dashboard")
def get_dashboard() -> dict:
    """Return portfolio summary and merchant-level dashboard rows."""
    return {
        "portfolio_summary": generate_portfolio_summary(portfolio_decisions),
        "merchants": [_dashboard_merchant_item(decision) for decision in portfolio_decisions],
    }


@app.get("/ui", response_class=HTMLResponse)
def dashboard_ui(request: Request) -> HTMLResponse:
    """Render simple HTML dashboard."""
    context = {
        "request": request,
        "portfolio_summary": generate_portfolio_summary(portfolio_decisions),
        "merchants": [_dashboard_merchant_item(decision) for decision in portfolio_decisions],
    }
    return templates.TemplateResponse("dashboard.html", context)


def _get_decision_by_merchant_id(merchant_id: str) -> dict | None:
    """Return cached decision for merchant, if present."""
    for decision in portfolio_decisions:
        if decision.get("merchant_id") == merchant_id:
            return decision
    return None


@app.post("/accept-offer/{merchant_id}")
def accept_offer(merchant_id: str) -> dict:
    """Accept an approved merchant offer, persist acceptance, and return a mock mandate reference."""
    decision = _get_decision_by_merchant_id(merchant_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Merchant not found")

    offer = decision.get("offer", {})
    decision_status = "rejected" if isinstance(offer, dict) and offer.get("status") == "REJECTED" else "approved"
    if decision_status != "approved":
        raise HTTPException(
            status_code=400,
            detail="Offer cannot be accepted because merchant was not approved",
        )

    accepted_merchants.add(merchant_id)
    mandate_reference = f"MOCK-NACH-{random.randint(1000, 9999)}"
    try:
        save_accepted_offer(
            merchant_id=merchant_id,
            mode="grab_credit",
            mandate_reference=mandate_reference,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist accepted offer: {exc}",
        ) from exc

    return {
        "merchant_id": merchant_id,
        "status": "mandate_initiated",
        "mandate_reference": mandate_reference,
    }


@app.get("/portfolio/analytics")
def get_portfolio_analytics_endpoint(mode: str | None = None) -> dict:
    """Return portfolio analytics computed from persisted underwriting decisions."""
    try:
        return get_portfolio_analytics(mode)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch portfolio analytics: {exc}",
        ) from exc


@app.get("/portfolio/analytics/v2")
def get_portfolio_analytics_v2_endpoint(mode: str | None = None) -> dict:
    """Return SQL-aggregated portfolio analytics from persisted underwriting decisions."""
    try:
        return get_portfolio_analytics_v2(mode)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch portfolio analytics v2: {exc}",
        ) from exc


@app.get("/portfolio/risk-alerts")
def get_portfolio_risk_alerts_endpoint(mode: str | None = None) -> dict:
    """Return portfolio risk alerts derived from persisted underwriting decisions."""
    try:
        alerts = get_portfolio_risk_alerts(mode)
        return {
            "alert_count": len(alerts),
            "alerts": alerts,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch portfolio risk alerts: {exc}",
        ) from exc


@app.get("/decisions/{merchant_id}")
def get_merchant_decisions(
    merchant_id: str,
    mode: str | None = None,
    limit: int = 20,
    cursor: str | None = None,
) -> dict:
    """Return persisted underwriting decision history for a merchant from SQLite."""
    if limit > 100:
        raise HTTPException(status_code=422, detail="limit must be less than or equal to 100")
    if limit <= 0:
        raise HTTPException(status_code=422, detail="limit must be greater than zero")

    try:
        result = get_decision_history(merchant_id, mode=mode, limit=limit, cursor=cursor)
        decisions = result.get("data", [])
        if not decisions:
            raise HTTPException(status_code=404, detail="No decision history found for this merchant")
        return {
            "merchant_id": merchant_id,
            "total_records": len(decisions),
            "decisions": decisions,
            "next_cursor": result.get("next_cursor"),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch decision history: {exc}",
        ) from exc


@app.post("/underwrite")
def underwrite_merchant(payload: MerchantInput) -> dict:
    """Run credit underwriting for an input merchant and return decision + AI explanation."""
    try:
        merchant = MerchantStub(**payload.model_dump())
        decision = generate_underwriting_decision(
            merchant=merchant,
            mode="grab_credit",
            category_averages=BASELINE_CATEGORY_AVERAGES,
        )
        trail = generate_underwriting_decision_trail(
            merchant=merchant,
            category_averages=BASELINE_CATEGORY_AVERAGES,
        )

        offer = decision.get("offer", {})
        tier = offer.get("tier") if isinstance(offer, dict) else None

        # Persist underwriting result to SQLite
        upsert_merchant(merchant.merchant_id, merchant.category)
        save_underwriting_decision(
            merchant_id=merchant.merchant_id,
            mode="grab_credit",
            risk_score=decision.get("risk_score"),
            tier=tier,
            confidence_level=decision.get("confidence_level"),
            offer=offer,
            ai_explanation=trail.get("final_explanation"),
        )

        return {
            "merchant_id": decision.get("merchant_id"),
            "risk_score": decision.get("risk_score"),
            "tier": tier,
            "confidence_level": decision.get("confidence_level"),
            "offer": offer,
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
