from app.demo_data import build_demo_merchants
from app.explainability import CategoryAverages
from app.underwriting import generate_underwriting_decision

BASELINE = CategoryAverages(refund_rate=4.5, customer_return_rate=60.0, yoy_gmv_growth=10.0)


def _merchant(merchant_id: str):
    for merchant in build_demo_merchants():
        if merchant.merchant_id == merchant_id:
            return merchant
    raise AssertionError(f"Merchant {merchant_id} not found")


def test_credit_rejection_for_high_refund() -> None:
    merchant = _merchant("M-HOME-REJ")
    decision = generate_underwriting_decision(merchant=merchant, mode="grab_credit", category_averages=BASELINE)
    assert decision["offer"]["status"] == "REJECTED"


def test_insurance_mode_outputs_policy_fields() -> None:
    merchant = _merchant("M-FASHION-T1")
    decision = generate_underwriting_decision(merchant=merchant, mode="grab_insurance", category_averages=BASELINE)
    offer = decision["offer"]
    assert offer["mode"] == "grab_insurance"
    assert offer["policy_type"] == "Comprehensive Coverage"
    assert isinstance(offer["coverage_amount_lakhs"], float)
    assert isinstance(offer["premium_quote_pct"], float)


def test_insurance_seasonality_adjustment_applies() -> None:
    merchant = _merchant("M-EDU-T2")  # seasonality > 2.0
    decision = generate_underwriting_decision(merchant=merchant, mode="grab_insurance", category_averages=BASELINE)
    offer = decision["offer"]
    breakdown = offer["financial_breakdown"]["computed_values"]
    assert breakdown["seasonality_reduction_applied"] is True
