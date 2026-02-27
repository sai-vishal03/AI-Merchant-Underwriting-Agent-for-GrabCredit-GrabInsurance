"""Manual portfolio test harness for underwriting + portfolio summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.explainability import CategoryAverages, generate_underwriting_decision_trail
from app.portfolio import generate_portfolio_summary
from app.server import MerchantStub, build_demo_merchants, build_offer_message
from app.underwriting import generate_underwriting_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Run underwriting harness on sample merchants")
    parser.add_argument("--mode", choices=["grab_credit", "grab_insurance"], default="grab_credit")
    parser.add_argument(
        "--send-demo",
        action="store_true",
        help="Print WhatsApp message simulation for approved merchants",
    )
    args = parser.parse_args()
    mode = args.mode

    baseline = CategoryAverages(refund_rate=4.5, customer_return_rate=60.0, yoy_gmv_growth=10.0)

    decisions: list[dict] = []
    merchants = build_demo_merchants()

    for merchant in merchants:
        decision = generate_underwriting_decision(merchant=merchant, mode=mode, category_averages=baseline)
        trail = generate_underwriting_decision_trail(merchant=merchant, category_averages=baseline, mode=mode)
        decision["ai_explanation"] = trail.get("final_explanation", "")
        decisions.append(decision)

        offer = decision.get("offer", {})
        status = "REJECTED" if offer.get("status") == "REJECTED" else "APPROVED"
        tier = offer.get("tier", "N/A")
        print(
            f"merchant_id={decision.get('merchant_id')} | "
            f"category={merchant.category} | "
            f"tier={tier} | "
            f"confidence={decision.get('confidence_level')} | "
            f"status={status}"
        )

        if args.send_demo and status == "APPROVED":
            if not merchant.phone_number:
                print(f"  [send-demo] skipped {merchant.merchant_id}: missing phone_number")
                continue
            message = build_offer_message(decision=decision, ai_explanation=str(decision.get("ai_explanation", "")))
            print(f"  [send-demo] to={merchant.phone_number}")
            print(f"  [send-demo] message:\n{message}\n")

    summary = generate_portfolio_summary(decisions)
    print(f"\nPortfolio Summary ({mode})")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
