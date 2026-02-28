# AI Merchant Underwriting Platform (GrabCredit + GrabInsurance)

A production-grade, deterministic underwriting platform built on **FastAPI** with a **multi-agent decisioning architecture**, model version controls, governance-grade audit snapshots, replay guarantees, and an analytics dashboard.

---

## Architecture Overview

The platform separates decisioning, persistence, API orchestration, and UI analytics into modular components.

### Core Layers

1. **API Layer (`app/server.py`)**
   - FastAPI endpoints for underwriting, replay, dashboard analytics, feature flags, health, and simulation.
   - Request timing middleware, structured logging, in-memory rate limiting, and graceful error handling.

2. **Agent Layer (`app/agent.py`)**
   - `FeatureEngineeringAgent`
   - `RiskAssessmentAgent`
   - `OfferGenerationAgent`
   - `ExplainabilityAgent`
   - `MerchantUnderwritingOrchestrator` (public orchestration entrypoint)

3. **Persistence Layer (`app/db.py`)**
   - SQLite-backed model versioning.
   - Idempotent underwriting persistence by `request_hash`.
   - Audit snapshot store with JSON integrity checks.
   - Dashboard analytics and decision-history query helpers.

4. **Dashboard Layer (`templates/dashboard.html`)**
   - Decision history v2 with pagination and filters.
   - Snapshot modal and replay action.
   - Risk analytics visualizations (Chart.js).

---

## Multi-Agent Design

### 1) FeatureEngineeringAgent
Computes deterministic derived features from merchant profile:
- `avg_gmv`
- `yoy_growth_percent`
- `volatility_index`
- `refund_risk_score`
- `loyalty_score`

Supports additional strict checks under `strict_validation` feature flag.

### 2) RiskAssessmentAgent
Applies nonlinear weighted risk matrix:
- Growth contribution
- Volatility contribution
- Refund contribution
- Loyalty contribution

Returns:
- `risk_score` (0-100)
- `tier`
- `rejection_reason` (if any)
- `risk_components`
- `dominant_component`

### 3) OfferGenerationAgent
Builds mode-specific approved/rejected offers.
- `grab_credit`: credit limit, interest rate, tenure options.
- `grab_insurance`: coverage, premium, policy type.

### 4) ExplainabilityAgent
Generates dynamic rationale tied to live metrics and dominant risk movement.

### 5) MerchantUnderwritingOrchestrator
Coordinates all agents and:
- Resolves active model
- Generates deterministic `request_hash`
- Persists decision + audit snapshot (unless `simulation_mode=True`)

---

## Risk Model Logic

Risk is computed from nonlinear bucketed contributions:
- Growth buckets
- Volatility buckets
- Refund-rate buckets
- Loyalty buckets

Final risk score:
- Clamped between 0 and 100
- Deterministic for identical input payload

Tiering and rejection logic operate on risk score plus policy guardrails.

---

## Deterministic Replay Guarantee

Endpoint: `POST /replay/{request_hash}`

Workflow:
1. Fetches stored request payload by hash.
2. Re-runs underwriting in `simulation_mode=True` (no writes).
3. Compares **risk score**, **tier**, and **model_id**.

Replay response includes parity result and both stored/replayed values.

---

## Feature Flag System

Available flags:
- `explainability_v2`
- `strict_validation`
- `simulation_mode`

Endpoints:
- `GET /feature-flags`
- `POST /feature-flags/{flag}?enabled=true|false`

---

## Model Versioning

`models` table fields:
- `id`
- `name`
- `version`
- `is_active`
- `created_at`

Supported helpers:
- `create_model_version(name, version)`
- `activate_model(model_id)`
- `get_active_model()`

Underwriting decisions store `model_id` for traceability and replay parity.

---

## Simulation Engine

Endpoint: `POST /simulate-portfolio`

Produces deterministic summary over synthetic merchants:
- total
- approvals
- rejections
- tier distribution
- average risk score
- dominant component breakdown

---


## Product Mode Support

Underwriting supports both product modes:
- `grab_credit`
- `grab_insurance`

Credit offers return `credit_limit_lakhs`, `interest_rate_tier`, and `tenure_options`.
Insurance offers return `coverage_amount`, `premium_quote`, and `suggested_policy_type`.

## WhatsApp Offer Integration

`app/whatsapp.py` provides `send_whatsapp_offer(phone_number, message)` using Twilio sandbox environment variables:
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_WHATSAPP_NUMBER`

When `phone_number` exists on merchant payload and offer is approved, the orchestrator attempts to send the offer notification.

## Accept Offer Flow

Endpoint: `POST /accept-offer/{request_hash}`
- Marks offer status as accepted
- Logs acceptance audit snapshot
- Simulates NACH mandate creation

## Dashboard

Route: `GET /dashboard`

Includes:
- Summary cards (decisions, approvals, rejection rate, avg risk)
- Dominant component pie chart
- Risk component average bar chart
- Tier trend line chart
- Decision history v2 table:
  - pagination
  - tier filter
  - model version filter
  - replay action
  - snapshot modal

Additional endpoints:
- `GET /dashboard/analytics`
- `GET /dashboard/decisions?page=1&limit=20&tier=Tier 1&model_id=...`
- `GET /dashboard/snapshot/{request_hash}`

---

## How to Run

### 1) Install dependencies
```bash
pip install fastapi uvicorn jinja2 pydantic
```

### 2) Start API
```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### 3) Access
- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8000/dashboard`
- Health: `http://localhost:8000/health`

---

## API Grouping in OpenAPI

Routes are tagged for production discoverability:
- **Underwriting**
- **Replay**
- **Dashboard**
- **Feature Flags**
- **System**

`/underwrite` includes documented response models, request examples, and error response schemas.

---

## Future Improvements

- Durable distributed rate limiting (e.g., Redis).
- Authentication + RBAC for feature flags and model activation.
- Online drift monitoring and automated rollback hooks.
- Asynchronous audit pipelines and data warehouse export.
- End-to-end integration tests for replay and dashboard APIs.

---

## Final Architecture Diagram (Text)

```text
Client/UI
  ├─ /dashboard (HTML + charts + decision history)
  ├─ /underwrite
  ├─ /replay/{request_hash}
  ├─ /simulate-portfolio
  └─ /feature-flags/*

FastAPI API Layer (server.py)
  ├─ middleware: rate limit + timing
  ├─ structured JSON logging
  ├─ graceful global exception handler
  └─ tagged routes (Underwriting, Replay, Dashboard, Feature Flags, System)

Orchestration Layer (agent.py)
  ├─ FeatureEngineeringAgent
  ├─ RiskAssessmentAgent
  ├─ OfferGenerationAgent
  ├─ ExplainabilityAgent
  └─ MerchantUnderwritingOrchestrator
       ├─ deterministic request_hash
       ├─ model resolution
       └─ persistence + audit hooks

Persistence Layer (db.py / SQLite)
  ├─ models
  ├─ underwriting_decisions (idempotent by request_hash)
  ├─ decision_audit_trail (json_valid enforced snapshots)
  └─ dashboard analytics + history query helpers
```
