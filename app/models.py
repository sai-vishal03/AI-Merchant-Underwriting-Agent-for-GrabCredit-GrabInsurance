"""Data models for merchant underwriting inputs and derived analytics."""

from __future__ import annotations

from statistics import stdev

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MerchantProfile(BaseModel):
    """Represents a merchant profile used for underwriting decisions.

    The model enforces strong validation for all input fields and provides
    utility methods to compute key risk and growth indicators derived from
    monthly GMV history.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    merchant_id: str = Field(..., min_length=1, description="Unique merchant identifier.")
    category: str = Field(..., min_length=1, description="Merchant business category.")
    monthly_gmv_12m: list[float] = Field(
        ...,
        min_length=12,
        max_length=12,
        description="Merchant gross merchandise value for each of the last 12 months.",
    )
    coupon_redemption_rate: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of issued coupons redeemed by customers.",
    )
    unique_customer_count: int = Field(
        ...,
        ge=0,
        description="Total count of unique customers in the evaluation period.",
    )
    customer_return_rate: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of customers who placed repeat orders.",
    )
    avg_order_value: float = Field(
        ...,
        ge=0,
        description="Average order value in INR.",
    )
    seasonality_index: float = Field(
        ...,
        ge=0,
        description="Peak-to-trough GMV ratio representing demand seasonality.",
    )
    deal_exclusivity_rate: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of deals that are platform-exclusive.",
    )
    return_and_refund_rate: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of transactions that are returned or refunded.",
    )

    @field_validator("monthly_gmv_12m")
    @classmethod
    def validate_monthly_gmv_values(cls, values: list[float]) -> list[float]:
        """Validate that monthly GMV values are non-negative."""
        if any(value < 0 for value in values):
            raise ValueError("monthly_gmv_12m must contain only non-negative values")
        return values

    def compute_yoy_gmv_growth(self) -> float:
        """Compute YoY GMV growth (%) using first-to-last month comparison.

        Returns:
            Percentage growth from month 1 to month 12.

        Raises:
            ValueError: If the first month GMV is zero, growth is undefined.
        """
        start_gmv = self.monthly_gmv_12m[0]
        end_gmv = self.monthly_gmv_12m[-1]

        if start_gmv == 0:
            raise ValueError("Cannot compute YoY growth when first month GMV is 0")

        return ((end_gmv - start_gmv) / start_gmv) * 100

    def compute_gmv_volatility(self) -> float:
        """Compute GMV volatility as sample standard deviation across 12 months."""
        if len(self.monthly_gmv_12m) < 2:
            return 0.0
        return stdev(self.monthly_gmv_12m)
