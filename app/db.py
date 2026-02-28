"""SQLite-backed persistence helpers for underwriting agent flows."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).resolve().parents[1] / "underwriting.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _initialize_db() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS underwriting_decisions (
                id INTEGER PRIMARY KEY,
                merchant_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                model_id TEXT,
                risk_score INTEGER NOT NULL,
                tier TEXT NOT NULL,
                confidence_level TEXT NOT NULL,
                offer TEXT NOT NULL,
                ai_explanation TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                request_payload TEXT,
                decision_source TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (merchant_id, mode, request_hash)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_audit_trail (
                id INTEGER PRIMARY KEY,
                merchant_id TEXT,
                mode TEXT,
                model_name TEXT,
                model_version TEXT,
                request_hash TEXT,
                full_snapshot_json TEXT NOT NULL CHECK (json_valid(full_snapshot_json)),
                created_at TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_decision_audit_trail_mode_merchant_created_at
            ON decision_audit_trail (mode, merchant_id, created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_underwriting_decisions_created_at
            ON underwriting_decisions (created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_underwriting_decisions_tier_model_created
            ON underwriting_decisions (tier, model_id, created_at DESC)
            """
        )

        _ensure_column(conn, "underwriting_decisions", "model_id", "TEXT")
        _ensure_column(conn, "underwriting_decisions", "request_payload", "TEXT")

        active_count = conn.execute("SELECT COUNT(*) AS count FROM models WHERE is_active = 1").fetchone()["count"]
        if active_count == 0:
            first_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO models (id, name, version, is_active, created_at) VALUES (?, ?, ?, ?, ?)",
                (first_id, "underwriting-foundation", "v1", 1, datetime.now(timezone.utc).isoformat()),
            )


_initialize_db()


def create_model_version(name: str, version: str) -> dict[str, Any]:
    """Create a model version record in inactive state."""
    model_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO models (id, name, version, is_active, created_at) VALUES (?, ?, ?, ?, ?)",
            (model_id, name, version, 0, created_at),
        )
    return {"id": model_id, "name": name, "version": version, "is_active": False, "created_at": created_at}


def activate_model(model_id: str) -> bool:
    """Activate a model by id and deactivate all others."""
    with _get_connection() as conn:
        exists = conn.execute("SELECT id FROM models WHERE id = ? LIMIT 1", (model_id,)).fetchone()
        if not exists:
            return False
        conn.execute("UPDATE models SET is_active = 0")
        conn.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
    return True


def list_model_versions() -> list[dict[str, Any]]:
    """Return all model versions for dashboard filters."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, version, is_active, created_at FROM models ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "version": row["version"],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_active_model(mode: str | None = None) -> dict[str, Any] | None:
    """Return active model metadata."""
    del mode
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, version, is_active, created_at
            FROM models
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "model_id": row["id"],
        "model_name": row["name"],
        "model_version": row["version"],
        "active": bool(row["is_active"]),
        "created_at": row["created_at"],
    }


def save_underwriting_decision(
    *,
    merchant_id: str,
    mode: str,
    model_id: str | None,
    risk_score: int,
    tier: str,
    confidence_level: str,
    offer: dict[str, Any],
    ai_explanation: str,
    request_hash: str,
    request_payload: dict[str, Any] | None,
    decision_source: str,
    created_by: str,
) -> tuple[bool, dict[str, Any]]:
    """Persist underwriting decision using request-hash idempotency."""
    with _get_connection() as conn:
        existing = conn.execute(
            """
            SELECT merchant_id, mode, model_id, risk_score, tier, confidence_level, offer, ai_explanation,
                   request_hash, request_payload, decision_source, created_by, created_at
            FROM underwriting_decisions
            WHERE merchant_id = ? AND mode = ? AND request_hash = ?
            LIMIT 1
            """,
            (merchant_id, mode, request_hash),
        ).fetchone()
        if existing:
            return False, {
                "merchant_id": existing["merchant_id"],
                "mode": existing["mode"],
                "model_id": existing["model_id"],
                "risk_score": existing["risk_score"],
                "tier": existing["tier"],
                "confidence_level": existing["confidence_level"],
                "offer": json.loads(existing["offer"]),
                "ai_explanation": existing["ai_explanation"],
                "request_hash": existing["request_hash"],
                "request_payload": json.loads(existing["request_payload"]) if existing["request_payload"] else None,
                "decision_source": existing["decision_source"],
                "created_by": existing["created_by"],
                "created_at": existing["created_at"],
            }

        created_at = datetime.now(timezone.utc).isoformat()
        offer_json = json.dumps(offer, sort_keys=True)
        payload_json = json.dumps(request_payload, sort_keys=True) if request_payload is not None else None
        conn.execute(
            """
            INSERT INTO underwriting_decisions (
                merchant_id, mode, model_id, risk_score, tier, confidence_level, offer, ai_explanation,
                request_hash, request_payload, decision_source, created_by, created_at
            )
            SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            WHERE json_valid(?) AND (? IS NULL OR json_valid(?))
            """,
            (
                merchant_id,
                mode,
                model_id,
                risk_score,
                tier,
                confidence_level,
                offer_json,
                ai_explanation,
                request_hash,
                payload_json,
                decision_source,
                created_by,
                created_at,
                offer_json,
                payload_json,
                payload_json,
            ),
        )

    return True, {
        "merchant_id": merchant_id,
        "mode": mode,
        "model_id": model_id,
        "risk_score": risk_score,
        "tier": tier,
        "confidence_level": confidence_level,
        "offer": offer,
        "ai_explanation": ai_explanation,
        "request_hash": request_hash,
        "request_payload": request_payload,
        "decision_source": decision_source,
        "created_by": created_by,
        "created_at": created_at,
    }


def save_decision_audit_trail(
    *,
    merchant_id: str,
    mode: str,
    model_name: str,
    model_version: str,
    request_hash: str,
    full_snapshot: dict[str, Any],
) -> None:
    """Persist expanded audit trail snapshot for governance and review."""
    snapshot_json = json.dumps(full_snapshot, sort_keys=True)
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO decision_audit_trail (
                merchant_id, mode, model_name, model_version, request_hash, full_snapshot_json, created_at
            )
            SELECT ?, ?, ?, ?, ?, ?, ?
            WHERE json_valid(?)
            """,
            (
                merchant_id,
                mode,
                model_name,
                model_version,
                request_hash,
                snapshot_json,
                datetime.now(timezone.utc).isoformat(),
                snapshot_json,
            ),
        )


def get_decision_by_request_hash(request_hash: str) -> dict[str, Any] | None:
    """Fetch persisted underwriting decision by request hash."""
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT merchant_id, mode, model_id, risk_score, tier, confidence_level,
                   offer, ai_explanation, request_hash, request_payload, decision_source, created_by, created_at
            FROM underwriting_decisions
            WHERE request_hash = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (request_hash,),
        ).fetchone()
    if row is None:
        return None

    return {
        "merchant_id": row["merchant_id"],
        "mode": row["mode"],
        "model_id": row["model_id"],
        "risk_score": int(row["risk_score"]),
        "tier": row["tier"],
        "confidence_level": row["confidence_level"],
        "offer": json.loads(row["offer"]),
        "ai_explanation": row["ai_explanation"],
        "request_hash": row["request_hash"],
        "request_payload": json.loads(row["request_payload"]) if row["request_payload"] else None,
        "decision_source": row["decision_source"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
    }


def get_audit_snapshot(request_hash: str) -> dict[str, Any] | None:
    """Return full audit snapshot by request hash."""
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT request_hash, full_snapshot_json, created_at
            FROM decision_audit_trail
            WHERE request_hash = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (request_hash,),
        ).fetchone()
    if row is None:
        return None
    return {
        "request_hash": row["request_hash"],
        "snapshot": json.loads(row["full_snapshot_json"]),
        "created_at": row["created_at"],
    }


def get_dashboard_decisions(
    *,
    page: int = 1,
    limit: int = 20,
    tier: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Return paginated decision history with filters."""
    page = max(1, page)
    limit = max(1, min(limit, 100))
    offset = (page - 1) * limit

    conditions: list[str] = []
    params: list[Any] = []
    if tier:
        conditions.append("d.tier = ?")
        params.append(tier)
    if model_id:
        conditions.append("d.model_id = ?")
        params.append(model_id)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with _get_connection() as conn:
        count_row = conn.execute(
            f"SELECT COUNT(*) AS total FROM underwriting_decisions d {where_clause}",
            tuple(params),
        ).fetchone()

        rows = conn.execute(
            f"""
            SELECT d.merchant_id, d.model_id, d.mode, d.risk_score, d.tier, d.request_hash, d.created_at, d.offer,
                   a.full_snapshot_json
            FROM underwriting_decisions d
            LEFT JOIN (
                SELECT request_hash, MAX(created_at) AS latest_created
                FROM decision_audit_trail
                GROUP BY request_hash
            ) latest_audit ON latest_audit.request_hash = d.request_hash
            LEFT JOIN decision_audit_trail a
                ON a.request_hash = latest_audit.request_hash
                AND a.created_at = latest_audit.latest_created
            {where_clause}
            ORDER BY d.created_at DESC
            LIMIT ? OFFSET ?
            """,
            tuple([*params, limit, offset]),
        ).fetchall()

        model_ids = conn.execute(
            "SELECT DISTINCT model_id FROM underwriting_decisions WHERE model_id IS NOT NULL ORDER BY model_id"
        ).fetchall()

    decisions = []
    for row in rows:
        snapshot = json.loads(row["full_snapshot_json"]) if row["full_snapshot_json"] else {}
        offer = json.loads(row["offer"])
        offer_status = offer.get("offer_status")
        if not offer_status:
            offer_status = "rejected" if offer.get("status") == "REJECTED" else "pending"
        decisions.append(
            {
                "merchant_id": row["merchant_id"],
                "model_id": row["model_id"],
                "product_mode": row["mode"] if "mode" in row.keys() else None,
                "risk_score": int(row["risk_score"]),
                "tier": row["tier"],
                "dominant_component": snapshot.get("dominant_component") or snapshot.get("risk_snapshot", {}).get("dominant_component"),
                "approved": offer.get("status") == "APPROVED",
                "credit_limit_lakhs": offer.get("credit_limit_lakhs"),
                "coverage_amount": offer.get("coverage_amount"),
                "premium_quote": offer.get("premium_quote"),
                "offer_status": offer_status,
                "created_at": row["created_at"],
                "request_hash": row["request_hash"],
            }
        )

    total = int(count_row["total"]) if count_row else 0
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "decisions": decisions,
        "models": [row["model_id"] for row in model_ids if row["model_id"]],
    }


def get_dashboard_analytics() -> dict[str, Any]:
    """Return advanced risk analytics payload for dashboard v2."""
    with _get_connection() as conn:
        decisions = conn.execute(
            "SELECT tier, risk_score, created_at FROM underwriting_decisions ORDER BY created_at"
        ).fetchall()
        snapshots = conn.execute(
            "SELECT full_snapshot_json, created_at FROM decision_audit_trail ORDER BY created_at"
        ).fetchall()

    total = len(decisions)
    avg_risk_score = round(sum(int(row["risk_score"]) for row in decisions) / total, 2) if total else 0.0

    tier_distribution: dict[str, int] = {"Tier 1": 0, "Tier 2": 0, "Tier 3": 0}
    tier_trend_bucket: dict[str, dict[str, int]] = {}
    for row in decisions:
        tier = row["tier"]
        tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        day = str(row["created_at"])[:10]
        tier_trend_bucket.setdefault(day, {"Tier 1": 0, "Tier 2": 0, "Tier 3": 0})
        tier_trend_bucket[day][tier] = tier_trend_bucket[day].get(tier, 0) + 1

    dominant_component_distribution: dict[str, int] = {}
    component_sums: dict[str, float] = {"growth": 0.0, "volatility": 0.0, "refund": 0.0, "loyalty": 0.0}
    component_count = 0

    for row in snapshots:
        snapshot = json.loads(row["full_snapshot_json"])
        dominant = snapshot.get("dominant_component")
        if dominant:
            dominant_component_distribution[str(dominant)] = dominant_component_distribution.get(str(dominant), 0) + 1

        risk_components = snapshot.get("risk_components", {})
        if risk_components:
            component_count += 1
            for key in component_sums:
                component_sums[key] += float(risk_components.get(key, 0.0))

    risk_component_averages = {
        key: round((value / component_count), 2) if component_count else 0.0 for key, value in component_sums.items()
    }

    tier_trend = [
        {
            "date": date,
            "Tier 1": counts.get("Tier 1", 0),
            "Tier 2": counts.get("Tier 2", 0),
            "Tier 3": counts.get("Tier 3", 0),
        }
        for date, counts in sorted(tier_trend_bucket.items())
    ]

    return {
        "avg_risk_score": avg_risk_score,
        "tier_distribution": tier_distribution,
        "dominant_component_distribution": dominant_component_distribution,
        "risk_component_averages": risk_component_averages,
        "tier_trend": tier_trend,
    }


def get_dashboard_metrics() -> dict[str, Any]:
    """Aggregate dashboard metrics from persisted decisions and audit snapshots."""
    analytics = get_dashboard_analytics()
    total_decisions = sum(analytics["tier_distribution"].values())
    approvals = total_decisions

    with _get_connection() as conn:
        offers = conn.execute("SELECT offer FROM underwriting_decisions").fetchall()

    approvals = 0
    for row in offers:
        offer = json.loads(row["offer"])
        if offer.get("status") == "APPROVED":
            approvals += 1

    rejections = total_decisions - approvals
    approval_rate = round((approvals / total_decisions) * 100, 2) if total_decisions else 0.0

    return {
        "total_decisions": total_decisions,
        "approvals": approvals,
        "rejections": rejections,
        "approval_rate": approval_rate,
        "tier_distribution": analytics["tier_distribution"],
        "avg_risk_score": analytics["avg_risk_score"],
        "dominant_component_breakdown": analytics["dominant_component_distribution"],
    }



def mark_decision_accepted(request_hash: str) -> dict[str, Any] | None:
    """Mark offer status as accepted for a decision request hash."""
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT merchant_id, mode, model_id, risk_score, tier, confidence_level, offer,
                   ai_explanation, request_hash, request_payload, decision_source, created_by, created_at
            FROM underwriting_decisions
            WHERE request_hash = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (request_hash,),
        ).fetchone()
        if row is None:
            return None

        offer = json.loads(row["offer"])
        if offer.get("status") == "REJECTED":
            return {"request_hash": request_hash, "accepted": False, "reason": "rejected_offer"}

        offer["offer_status"] = "accepted"
        updated_offer = json.dumps(offer, sort_keys=True)
        conn.execute(
            "UPDATE underwriting_decisions SET offer = ? WHERE request_hash = ?",
            (updated_offer, request_hash),
        )

    return {
        "request_hash": request_hash,
        "merchant_id": row["merchant_id"],
        "mode": row["mode"],
        "model_id": row["model_id"],
        "tier": row["tier"],
        "risk_score": int(row["risk_score"]),
        "offer": offer,
        "accepted": True,
    }
