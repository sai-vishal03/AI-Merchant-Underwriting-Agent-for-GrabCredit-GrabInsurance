"""Lightweight FastAPI server for underwriting dashboard flows."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.explainability import CategoryAverages
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


app = FastAPI(title="Merchant Underwriting Dashboard API")

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))
merchants = _build_merchants()
portfolio_decisions: list[dict] = []
accepted_merchants: set[str] = set()


@app.on_event("startup")
def startup_load_decisions() -> None:
    """Generate and cache underwriting decisions at service startup."""
    portfolio_decisions.clear()
    for merchant in merchants:
        decision = generate_underwriting_decision(
            merchant=merchant,
            mode="grab_credit",
            category_averages=BASELINE_CATEGORY_AVERAGES,
        )
        portfolio_decisions.append(decision)


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


@app.post("/accept-offer/{merchant_id}")
def accept_offer(merchant_id: str) -> dict:
    """Mark merchant offer as accepted and return mock NACH initiation payload."""
    merchant_exists = any(decision.get("merchant_id") == merchant_id for decision in portfolio_decisions)
    if not merchant_exists:
        raise HTTPException(status_code=404, detail="Merchant not found")

    accepted_merchants.add(merchant_id)
    return {
        "merchant_id": merchant_id,
        "status": "mandate_initiated",
        "mandate_reference": f"MOCK-NACH-{random.randint(1000, 9999)}",
    }
