"""Manual portfolio test harness for underwriting + portfolio summary.

Creates representative edge-case merchant stubs, generates credit decisions,
and prints both per-merchant outcomes and aggregate portfolio summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from statistics import stdev

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.explainability import CategoryAverages
from app.portfolio import generate_portfolio_summary
from app.underwriting import generate_underwriting_decision


@dataclass
class MerchantStub:
    """Lightweight merchant object implementing the underwriting protocol."""

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
        """Compute YoY growth using first and last month GMV."""
        start = self.monthly_gmv_12m[0]
        end = self.monthly_gmv_12m[-1]
        if start == 0:
            raise ValueError("Cannot compute YoY growth when baseline GMV is zero")
        return ((end - start) / start) * 100

    def compute_gmv_volatility(self) -> float:
        """Compute sample standard deviation over monthly GMV series."""
        return stdev(self.monthly_gmv_12m)


def build_merchants() -> list[MerchantStub]:
    """Create six scenario merchants for portfolio stress coverage."""
    return [
        # 1) Strong Tier 1: high growth, low refunds.
        MerchantStub(
            merchant_id="M-T1-STRONG",
            category="fashion",
            monthly_gmv_12m=[1_000_000, 1_050_000, 1_120_000, 1_200_000, 1_280_000, 1_350_000, 1_420_000, 1_500_000, 1_620_000, 1_780_000, 1_920_000, 2_200_000],
            coupon_redemption_rate=22.0,
            unique_customer_count=8_500,
            customer_return_rate=74.0,
            avg_order_value=640.0,
            seasonality_index=1.25,
            deal_exclusivity_rate=32.0,
            return_and_refund_rate=2.1,
        ),
        # 2) Tier 2 haircut case: refunds above category average.
        MerchantStub(
            merchant_id="M-T2-HAIRCUT",
            category="grocery",
            monthly_gmv_12m=[1_800_000, 1_820_000, 1_790_000, 1_810_000, 1_850_000, 1_840_000, 1_860_000, 1_870_000, 1_880_000, 1_890_000, 1_900_000, 1_910_000],
            coupon_redemption_rate=35.0,
            unique_customer_count=4_200,
            customer_return_rate=63.0,
            avg_order_value=520.0,
            seasonality_index=1.20,
            deal_exclusivity_rate=40.0,
            return_and_refund_rate=5.2,
        ),
        # 3) Tier 3 with exposure cap: large ticket, low confidence, non-rejected.
        MerchantStub(
            merchant_id="M-T3-EXPOSURE-CAP",
            category="electronics",
            monthly_gmv_12m=[80_000_000, 81_500_000, 80_800_000, 82_000_000, 81_200_000, 82_500_000, 81_900_000, 83_200_000, 82_400_000, 83_500_000, 82_700_000, 83_000_000],
            coupon_redemption_rate=58.0,
            unique_customer_count=65_000,
            customer_return_rate=50.0,
            avg_order_value=2_300.0,
            seasonality_index=1.45,
            deal_exclusivity_rate=48.0,
            return_and_refund_rate=7.0,
        ),
        # 4) Hard rejection: refund >= 8%.
        MerchantStub(
            merchant_id="M-REJ-REFUND",
            category="home_decor",
            monthly_gmv_12m=[2_200_000, 2_150_000, 2_180_000, 2_160_000, 2_140_000, 2_130_000, 2_120_000, 2_100_000, 2_090_000, 2_080_000, 2_070_000, 2_060_000],
            coupon_redemption_rate=50.0,
            unique_customer_count=2_600,
            customer_return_rate=52.0,
            avg_order_value=700.0,
            seasonality_index=1.30,
            deal_exclusivity_rate=44.0,
            return_and_refund_rate=8.4,
        ),
        # 5) Negative growth rejection: YoY <= -10%.
        MerchantStub(
            merchant_id="M-REJ-NEG-GROWTH",
            category="travel",
            monthly_gmv_12m=[3_000_000, 2_850_000, 2_700_000, 2_650_000, 2_500_000, 2_400_000, 2_300_000, 2_200_000, 2_100_000, 2_000_000, 1_950_000, 1_850_000],
            coupon_redemption_rate=42.0,
            unique_customer_count=3_000,
            customer_return_rate=48.0,
            avg_order_value=1_050.0,
            seasonality_index=1.60,
            deal_exclusivity_rate=39.0,
            return_and_refund_rate=4.1,
        ),
        # 6) High-volatility seasonal case.
        MerchantStub(
            merchant_id="M-HIGH-VOL-SEASONAL",
            category="events",
            monthly_gmv_12m=[500_000, 520_000, 510_000, 530_000, 540_000, 520_000, 1_900_000, 2_100_000, 1_850_000, 560_000, 540_000, 530_000],
            coupon_redemption_rate=60.0,
            unique_customer_count=1_800,
            customer_return_rate=55.0,
            avg_order_value=1_300.0,
            seasonality_index=3.80,
            deal_exclusivity_rate=50.0,
            return_and_refund_rate=3.9,
        ),
    ]


def main() -> None:
    """Run scenario decisions and print merchant + portfolio summaries."""
    baseline = CategoryAverages(
        refund_rate=4.5,
        customer_return_rate=60.0,
        yoy_gmv_growth=10.0,
    )

    decisions: list[dict] = []
    for merchant in build_merchants():
        decision = generate_underwriting_decision(
            merchant=merchant,
            mode="grab_credit",
            category_averages=baseline,
        )
        decisions.append(decision)

        offer = decision.get("offer", {})
        status = "REJECTED" if offer.get("status") == "REJECTED" else "APPROVED"
        tier = offer.get("tier", "N/A")
        print(
            f"merchant_id={decision.get('merchant_id')} | "
            f"tier={tier} | "
            f"confidence={decision.get('confidence_level')} | "
            f"status={status}"
        )

    summary = generate_portfolio_summary(decisions)
    print("\nPortfolio Summary")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
