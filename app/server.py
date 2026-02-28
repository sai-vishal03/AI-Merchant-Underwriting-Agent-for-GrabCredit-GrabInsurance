"""Production-ready FastAPI server for underwriting, dashboard, and controls."""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev
from typing import Any, Deque

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from app.agent import (
    MerchantUnderwritingAgent,
    get_sample_merchants,
    run_portfolio_simulation,
    set_feature_flag,
)
from app.db import (
    get_audit_snapshot,
    get_dashboard_analytics,
    get_dashboard_decisions,
    get_dashboard_metrics,
    get_decision_by_request_hash,
    list_model_versions,
    mark_decision_accepted,
    save_decision_audit_trail,
)
from app.explainability import CategoryAverages
from app.underwriting import generate_underwriting_decision


class JsonLogFormatter(logging.Formatter):
    """Simple JSON log formatter for structured production logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload)


logger = logging.getLogger("underwriting_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ErrorResponse(BaseModel):
    detail: str


class UnderwriteResponse(BaseModel):
    merchant_id: str
    risk_score: int
    risk_contributions: dict[str, float]
    confidence_level: str
    offer: dict[str, Any]
    tier: str
    ai_explanation: str
    mode: str
    model_used: str
    model_id: str
    request_hash: str
    idempotent: bool


class ReplayResponse(BaseModel):
    request_hash: str
    matches: bool
    model_match: bool
    stored: dict[str, Any]
    replayed: dict[str, Any]


class DashboardDecisionsResponse(BaseModel):
    page: int
    limit: int
    total: int
    models: list[str]
    decisions: list[dict[str, Any]]


class SnapshotResponse(BaseModel):
    request_hash: str
    snapshot: dict[str, Any]
    created_at: str


class DashboardAnalyticsResponse(BaseModel):
    avg_risk_score: float
    tier_distribution: dict[str, int]
    dominant_component_distribution: dict[str, int]
    risk_component_averages: dict[str, float]
    tier_trend: list[dict[str, Any]]


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
    coupon_redemption_rate: float = Field(..., ge=0, le=100)
    unique_customer_count: int = Field(..., ge=0)
    customer_return_rate: float = Field(..., ge=0, le=100)
    avg_order_value: float = Field(..., ge=0)
    seasonality_index: float = Field(..., ge=0)
    deal_exclusivity_rate: float = Field(..., ge=0, le=100)
    return_and_refund_rate: float = Field(..., ge=0, le=100)
    phone_number: str | None = None
    mode: str = "grab_credit"

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


REQUEST_HISTORY: defaultdict[str, Deque[float]] = defaultdict(deque)
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60


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


@app.middleware("http")
async def rate_limit_and_timing(request: Request, call_next):
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    history = REQUEST_HISTORY[client_ip]
    while history and now - history[0] > RATE_LIMIT_WINDOW_SECONDS:
        history.popleft()
    if len(history) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    history.append(now)

    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    logger.info(
        f"request_completed {request.method} {request.url.path}",
        extra={"extra": {"path": request.url.path, "method": request.method, "status_code": response.status_code, "elapsed_ms": elapsed_ms}},
    )
    return response


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception):
    logger.error(
        f"unhandled_exception: {exc}",
        extra={"extra": {"path": request.url.path, "method": request.method}},
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
def startup_load_decisions() -> None:
    """Generate and cache underwriting decisions and load deterministic sample data."""
    portfolio_decisions.clear()
    for merchant in merchants:
        decision = generate_underwriting_decision(
            merchant=merchant,
            mode="grab_credit",
            category_averages=BASELINE_CATEGORY_AVERAGES,
        )
        portfolio_decisions.append(decision)

    # Deterministic sample loader for 10 merchants.
    seed_rows = get_sample_merchants()
    agent = MerchantUnderwritingAgent()
    for row in seed_rows:
        agent.evaluate_merchant(row, "grab_credit")




@app.get("/")
def root_redirect(request: Request):
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        return RedirectResponse(url="/dashboard", status_code=307)
    return {"status": "ok", "dashboard": "/dashboard"}

@app.get("/health", tags=["System"])
def health() -> dict:
    """Healthcheck endpoint."""
    return {"status": "ok"}


@app.get("/feature-flags", tags=["Feature Flags"])
def read_feature_flags() -> dict[str, bool]:
    """Get active feature flags."""
    return get_feature_flags()


@app.post("/feature-flags/{flag}", tags=["Feature Flags"], responses={422: {"model": ErrorResponse}})
def update_feature_flag(flag: str, enabled: bool = Query(..., description="Set to true/false")) -> dict[str, bool]:
    """Update a feature flag value."""
    try:
        return set_feature_flag(flag, enabled)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard(request: Request) -> HTMLResponse:
    """Render HTML dashboard with aggregated underwriting metrics."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "metrics": get_dashboard_metrics(),
            "models": list_model_versions(),
        },
    )


@app.get("/dashboard/data", tags=["Dashboard"])
def dashboard_data() -> dict:
    return get_dashboard_metrics()


@app.get("/dashboard/analytics", response_model=DashboardAnalyticsResponse, tags=["Dashboard"])
def dashboard_analytics() -> DashboardAnalyticsResponse:
    return DashboardAnalyticsResponse(**get_dashboard_analytics())


@app.get("/dashboard/decisions", response_model=DashboardDecisionsResponse, tags=["Dashboard"])
def dashboard_decisions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    tier: str | None = Query(None),
    model_id: str | None = Query(None),
) -> DashboardDecisionsResponse:
    return DashboardDecisionsResponse(**get_dashboard_decisions(page=page, limit=limit, tier=tier, model_id=model_id))


@app.get("/dashboard/snapshot/{request_hash}", response_model=SnapshotResponse, tags=["Dashboard"], responses={404: {"model": ErrorResponse}})
def dashboard_snapshot(request_hash: str) -> SnapshotResponse:
    snapshot = get_audit_snapshot(request_hash)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return SnapshotResponse(**snapshot)


@app.get("/ui", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard_ui(request: Request) -> HTMLResponse:
    return dashboard(request)


def _get_decision_by_merchant_id(merchant_id: str) -> dict | None:
    for decision in portfolio_decisions:
        if decision.get("merchant_id") == merchant_id:
            return decision
    return None


@app.post("/simulate-portfolio", tags=["Underwriting"], responses={422: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def simulate_portfolio(mode: str = Query("grab_credit")) -> dict:
    try:
        return run_portfolio_simulation(mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        if str(exc) == "No active model configured":
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=f"Failed to simulate portfolio: {exc}") from exc


@app.post("/replay/{request_hash}", response_model=ReplayResponse, tags=["Replay"], responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def replay_decision(request_hash: str) -> ReplayResponse:
    stored = get_decision_by_request_hash(request_hash)
    if stored is None or not stored.get("request_payload"):
        raise HTTPException(status_code=404, detail="Stored request not found")

    request_payload = stored["request_payload"]
    merchant_profile = request_payload.get("merchant_profile")
    mode = request_payload.get("mode")
    if not isinstance(merchant_profile, dict) or not isinstance(mode, str):
        raise HTTPException(status_code=422, detail="Stored request payload is malformed")

    replayed = MerchantUnderwritingAgent().evaluate_merchant(
        merchant_profile,
        mode,
        simulation_override=True,
    )

    score_match = replayed["risk_score"] == stored["risk_score"]
    tier_match = replayed["tier"] == stored["tier"]
    model_match = replayed.get("model_id") == stored.get("model_id")

    return ReplayResponse(
        request_hash=request_hash,
        matches=score_match and tier_match and model_match,
        model_match=model_match,
        stored={"risk_score": stored["risk_score"], "tier": stored["tier"], "model_id": stored.get("model_id")},
        replayed={"risk_score": replayed["risk_score"], "tier": replayed["tier"], "model_id": replayed.get("model_id")},
    )


@app.post("/accept-offer/{request_hash}", tags=["Underwriting"], responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def accept_offer(request_hash: str) -> dict:
    """Accept an approved offer, log audit, and simulate NACH creation."""
    result = mark_decision_accepted(request_hash)
    if result is None:
        # Backward-compatible path: treat value as legacy merchant_id from in-memory dashboard flow.
        decision = _get_decision_by_merchant_id(request_hash)
        if decision is None:
            raise HTTPException(status_code=404, detail="Decision not found")
        offer = decision.get("offer", {})
        if isinstance(offer, dict) and offer.get("status") == "REJECTED":
            raise HTTPException(status_code=422, detail="Offer cannot be accepted")
        accepted_merchants.add(request_hash)
        return {
            "merchant_id": request_hash,
            "status": "mandate_initiated",
            "mandate_reference": f"MOCK-NACH-{random.randint(1000, 9999)}",
        }

    if not result.get("accepted"):
        raise HTTPException(status_code=422, detail="Offer cannot be accepted")

    save_decision_audit_trail(
        merchant_id=str(result["merchant_id"]),
        mode=str(result["mode"]),
        model_name="accept_flow",
        model_version="v1",
        request_hash=request_hash,
        full_snapshot={
            "event": "offer_accepted",
            "merchant_id": result["merchant_id"],
            "mode": result["mode"],
            "model_id": result.get("model_id"),
            "offer": result.get("offer"),
            "nach_status": "mandate_created",
        },
    )

    return {
        "request_hash": request_hash,
        "merchant_id": result["merchant_id"],
        "status": "accepted",
        "nach_mandate_id": f"NACH-{random.randint(100000, 999999)}",
    }


@app.post("/underwrite", response_model=UnderwriteResponse, tags=["Underwriting"], responses={422: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def underwrite_merchant(
    payload: MerchantInput = Body(
        ...,
        example={
            "merchant_id": "M-DEMO-001",
            "category": "fashion",
            "monthly_gmv_12m": [1000000, 1020000, 1040000, 1060000, 1080000, 1100000, 1120000, 1140000, 1160000, 1180000, 1200000, 1220000],
            "coupon_redemption_rate": 20.0,
            "unique_customer_count": 5500,
            "customer_return_rate": 68.0,
            "avg_order_value": 600.0,
            "seasonality_index": 1.25,
            "deal_exclusivity_rate": 35.0,
            "return_and_refund_rate": 3.2,
            "phone_number": "+919999999111",
            "mode": "grab_credit",
        },
    ),
) -> UnderwriteResponse:
    try:
        result = MerchantUnderwritingAgent().evaluate_merchant(payload.model_dump(), payload.mode)
        return UnderwriteResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        if str(exc) == "No active model configured":
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=f"Failed to underwrite merchant: {exc}") from exc
    except HTTPException:
        raise
