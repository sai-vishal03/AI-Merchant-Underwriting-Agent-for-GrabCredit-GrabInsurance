"""Shared demo data builders for server and CLI harness.

This module deliberately avoids FastAPI imports so scripts can run in minimal
Python environments while reusing identical demo merchants and message formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import stdev


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
