# AI-Merchant-Underwriting-Agent-for-GrabCredit-GrabInsurance

AI underwriting demo platform for evaluating merchant risk and generating explainable pre-approved offers for:
- **GrabCredit** (credit limits, interest rates, tenures)
- **GrabInsurance** (coverage, premium, policy type)

It includes:
- deterministic risk scoring + tiering,
- explainability trails with optional LLM rationale,
- portfolio-level analytics,
- FastAPI dashboard,
- WhatsApp offer dispatch (mock/Twilio provider modes).

## Quick start

```bash
python -m pip install fastapi uvicorn jinja2 httpx
python -m uvicorn app.server:app --reload
```

Open:
- `http://127.0.0.1:8000/ui`
- `http://127.0.0.1:8000/ui?mode=grab_insurance`

## Useful API endpoints

- `GET /health`
- `GET /dashboard?mode=grab_credit|grab_insurance`
- `GET /decision-trail/{merchant_id}?mode=...`
- `POST /send-offer/{merchant_id}?mode=...`
- `POST /accept-offer/{merchant_id}?mode=...`
- `POST /refresh-decisions?mode=...`
- `POST /underwrite?mode=...`

## Demo harness (no FastAPI import dependency)

```bash
python scripts/test_portfolio.py --mode grab_credit
python scripts/test_portfolio.py --mode grab_insurance --send-demo
```
