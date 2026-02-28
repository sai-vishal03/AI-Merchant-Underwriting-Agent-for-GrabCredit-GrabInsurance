"""SQLite persistence helpers for underwriting entities."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

DATABASE_FILE = "underwriting.db"


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection configured to return row objects."""
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    """Create persistence tables when they do not already exist."""
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS merchants (
                merchant_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS underwriting_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                merchant_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                risk_score REAL,
                tier TEXT,
                confidence_level TEXT,
                offer_json TEXT,
                ai_explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS accepted_offers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                merchant_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                mandate_reference TEXT NOT NULL,
                accepted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_underwriting_mode
            ON underwriting_decisions(mode)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_underwriting_created_id
            ON underwriting_decisions(created_at DESC, id DESC)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_underwriting_tier
            ON underwriting_decisions(tier)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_underwriting_merchant_id
            ON underwriting_decisions(merchant_id)
            """
        )
        connection.commit()
    finally:
        connection.close()


def upsert_merchant(merchant_id: str, category: str) -> None:
    """Insert a merchant row if it does not already exist."""
    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO merchants (merchant_id, category)
            VALUES (?, ?)
            """,
            (merchant_id, category),
        )
        connection.commit()
    finally:
        connection.close()


def save_underwriting_decision(
    merchant_id: str,
    mode: str,
    risk_score: float,
    tier: str | None,
    confidence_level: str,
    offer: dict[str, Any],
    ai_explanation: str,
) -> None:
    """Persist an underwriting decision row for a merchant."""
    connection = get_connection()
    try:
        offer_json = json.dumps(offer)
        connection.execute(
            """
            INSERT INTO underwriting_decisions (
                merchant_id,
                mode,
                risk_score,
                tier,
                confidence_level,
                offer_json,
                ai_explanation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                merchant_id,
                mode,
                risk_score,
                tier,
                confidence_level,
                offer_json,
                ai_explanation,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def save_accepted_offer(
    merchant_id: str,
    mode: str,
    mandate_reference: str,
) -> None:
    """Persist an accepted offer with mandate reference."""
    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT INTO accepted_offers (merchant_id, mode, mandate_reference)
            VALUES (?, ?, ?)
            """,
            (merchant_id, mode, mandate_reference),
        )
        connection.commit()
    finally:
        connection.close()


def get_accepted_merchant_ids() -> set[str]:
    """Return distinct merchant IDs with accepted offers persisted in SQLite."""
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT merchant_id FROM accepted_offers
            """
        ).fetchall()
        return {str(row["merchant_id"]) for row in rows}
    finally:
        connection.close()



def get_decision_history(
    merchant_id: str,
    mode: str | None = None,
    limit: int = 20,
    cursor: str | None = None,
) -> dict:
    """Return cursor-paginated underwriting history for a merchant."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if limit > 100:
        raise ValueError("limit must be less than or equal to 100")

    cursor_created_at: str | None = None
    cursor_id: int | None = None
    if cursor is not None:
        if "_" not in cursor:
            raise ValueError("Invalid cursor format")
        cursor_created_at, cursor_id_raw = cursor.rsplit("_", 1)
        if not cursor_created_at:
            raise ValueError("Invalid cursor format")
        try:
            cursor_id = int(cursor_id_raw)
        except ValueError as exc:
            raise ValueError("Invalid cursor format") from exc

    connection = get_connection()
    try:
        query = [
            "SELECT id, merchant_id, mode, risk_score, tier, confidence_level, offer_json, created_at",
            "FROM underwriting_decisions",
            "WHERE merchant_id = ?",
        ]
        params: list[str | int] = [merchant_id]

        if mode is not None:
            query.append("AND mode = ?")
            params.append(mode)

        if cursor_created_at is not None and cursor_id is not None:
            query.append("AND (created_at, id) < (?, ?)")
            params.extend([cursor_created_at, cursor_id])

        query.append("ORDER BY created_at DESC, id DESC")
        query.append("LIMIT ?")
        params.append(limit)

        rows = connection.execute("\n".join(query), tuple(params)).fetchall()
        data = [
            {
                "id": row["id"],
                "merchant_id": str(row["merchant_id"]),
                "mode": row["mode"],
                "risk_score": row["risk_score"],
                "tier": row["tier"],
                "confidence": row["confidence_level"],
                "offer_json": json.loads(row["offer_json"]) if row["offer_json"] else None,
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        next_cursor = None
        if data:
            last_row = data[-1]
            next_cursor = f"{last_row['created_at']}_{last_row['id']}"
        return {
            "data": data,
            "next_cursor": next_cursor,
        }
    finally:
        connection.close()



def get_portfolio_analytics(mode: str | None = None) -> dict:
    """Compute portfolio analytics from persisted underwriting decisions."""
    connection = get_connection()
    try:
        query = [
            "SELECT risk_score, tier, confidence_level, offer_json",
            "FROM underwriting_decisions",
        ]
        params: list[str] = []
        if mode is not None:
            query.append("WHERE mode = ?")
            params.append(mode)

        rows = connection.execute("\n".join(query), tuple(params)).fetchall()

        tier_distribution = {"Tier 1": 0, "Tier 2": 0, "Tier 3": 0}
        confidence_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        if not rows:
            return {
                "total_decisions": 0,
                "approval_count": 0,
                "rejection_count": 0,
                "approval_rate": 0.0,
                "average_risk_score": 0.0,
                "tier_distribution": tier_distribution,
                "confidence_distribution": confidence_distribution,
                "total_exposure_lakhs": 0.0,
            }

        total_decisions = len(rows)
        approval_count = 0
        rejection_count = 0
        total_exposure_lakhs = 0.0

        risk_scores = [float(row["risk_score"]) for row in rows if row["risk_score"] is not None]

        for row in rows:
            tier = row["tier"]
            if tier in tier_distribution:
                tier_distribution[tier] += 1

            confidence_level = (row["confidence_level"] or "").lower()
            if "high" in confidence_level:
                confidence_distribution["HIGH"] += 1
            elif "medium" in confidence_level:
                confidence_distribution["MEDIUM"] += 1
            elif "low" in confidence_level:
                confidence_distribution["LOW"] += 1

            offer_data: dict[str, object] = {}
            if row["offer_json"]:
                try:
                    parsed_offer = json.loads(row["offer_json"])
                    if isinstance(parsed_offer, dict):
                        offer_data = parsed_offer
                except (json.JSONDecodeError, TypeError):
                    offer_data = {}

            offer_status = offer_data.get("status")
            is_rejected = offer_status == "REJECTED"

            if is_rejected:
                rejection_count += 1
            else:
                approval_count += 1
                credit_limit_lakhs = offer_data.get("credit_limit_lakhs", 0)
                if isinstance(credit_limit_lakhs, (int, float)):
                    total_exposure_lakhs += float(credit_limit_lakhs)

        approval_rate = (approval_count / total_decisions) * 100 if total_decisions else 0.0
        average_risk_score = (sum(risk_scores) / len(risk_scores)) if risk_scores else 0.0

        return {
            "total_decisions": total_decisions,
            "approval_count": approval_count,
            "rejection_count": rejection_count,
            "approval_rate": round(approval_rate, 2),
            "average_risk_score": round(average_risk_score, 2),
            "tier_distribution": tier_distribution,
            "confidence_distribution": confidence_distribution,
            "total_exposure_lakhs": round(total_exposure_lakhs, 2),
        }
    finally:
        connection.close()



def get_portfolio_analytics_v2(mode: str | None = None) -> dict:
    """Compute SQL-aggregated portfolio analytics from persisted underwriting decisions."""
    connection = get_connection()
    try:
        where_clause = ""
        params: tuple[str, ...] = ()
        if mode is not None:
            where_clause = "WHERE mode = ?"
            params = (mode,)

        # Aggregate total number of persisted decisions.
        total_query = "SELECT COUNT(*) AS total_count FROM underwriting_decisions"
        if where_clause:
            total_query = f"{total_query} {where_clause}"
        total_decisions = int(connection.execute(total_query, params).fetchone()["total_count"])

        # Aggregate average risk score (SQLite AVG ignores NULL values).
        avg_query = "SELECT AVG(risk_score) AS average_risk_score FROM underwriting_decisions"
        if where_clause:
            avg_query = f"{avg_query} {where_clause}"
        average_risk_value = connection.execute(avg_query, params).fetchone()["average_risk_score"]

        # Aggregate tier distribution across persisted decisions.
        tier_distribution = {"Tier 1": 0, "Tier 2": 0, "Tier 3": 0}
        tier_query_parts = [
            "SELECT tier, COUNT(*) AS tier_count",
            "FROM underwriting_decisions",
        ]
        if where_clause:
            tier_query_parts.append(where_clause)
        tier_query_parts.append("GROUP BY tier")
        tier_rows = connection.execute("\n".join(tier_query_parts), params).fetchall()
        for row in tier_rows:
            tier = row["tier"]
            if tier in tier_distribution:
                tier_distribution[tier] = int(row["tier_count"])

        # Aggregate confidence distribution and normalize labels into HIGH/MEDIUM/LOW buckets.
        confidence_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        confidence_query_parts = [
            "SELECT confidence_level, COUNT(*) AS confidence_count",
            "FROM underwriting_decisions",
        ]
        if where_clause:
            confidence_query_parts.append(where_clause)
        confidence_query_parts.append("GROUP BY confidence_level")
        confidence_rows = connection.execute("\n".join(confidence_query_parts), params).fetchall()
        for row in confidence_rows:
            confidence_level = (row["confidence_level"] or "").lower()
            count = int(row["confidence_count"])
            if "high" in confidence_level:
                confidence_distribution["HIGH"] += count
            elif "medium" in confidence_level:
                confidence_distribution["MEDIUM"] += count
            elif "low" in confidence_level:
                confidence_distribution["LOW"] += count

        # Aggregate approval/rejection counts and total approved exposure using defensive JSON extraction.
        status_query_parts = [
            "SELECT",
            "    COALESCE(",
            "        SUM(",
            "            CASE",
            "                WHEN json_valid(offer_json)",
            "                     AND json_extract(offer_json, '$.status') = 'REJECTED'",
            "                THEN 1 ELSE 0",
            "            END",
            "        ),",
            "        0",
            "    ) AS rejection_count,",
            "    COALESCE(",
            "        SUM(",
            "            CASE",
            "                WHEN json_valid(offer_json)",
            "                     AND json_extract(offer_json, '$.status') IS NOT NULL",
            "                     AND json_extract(offer_json, '$.status') != 'REJECTED'",
            "                THEN 1 ELSE 0",
            "            END",
            "        ),",
            "        0",
            "    ) AS approval_count,",
            "    COALESCE(",
            "        SUM(",
            "            CASE",
            "                WHEN json_valid(offer_json)",
            "                     AND json_extract(offer_json, '$.status') IS NOT NULL",
            "                     AND json_extract(offer_json, '$.status') != 'REJECTED'",
            "                THEN COALESCE(CAST(json_extract(offer_json, '$.credit_limit_lakhs') AS REAL), 0)",
            "                ELSE 0",
            "            END",
            "        ),",
            "        0",
            "    ) AS total_exposure_lakhs",
            "FROM underwriting_decisions",
        ]
        if where_clause:
            status_query_parts.append(where_clause)
        status_agg_row = connection.execute("\n".join(status_query_parts), params).fetchone()

        approval_count = int(status_agg_row["approval_count"])
        rejection_count = int(status_agg_row["rejection_count"])
        total_exposure_lakhs = float(status_agg_row["total_exposure_lakhs"])

        approval_rate = (approval_count / total_decisions) * 100 if total_decisions else 0.0
        average_risk_score = float(average_risk_value) if average_risk_value is not None else 0.0

        return {
            "total_decisions": total_decisions,
            "approval_count": approval_count,
            "rejection_count": rejection_count,
            "approval_rate": round(approval_rate, 2),
            "average_risk_score": round(average_risk_score, 2),
            "tier_distribution": tier_distribution,
            "confidence_distribution": confidence_distribution,
            "total_exposure_lakhs": round(total_exposure_lakhs, 2),
        }
    finally:
        connection.close()



def get_portfolio_risk_alerts(
    mode: str | None = None,
    approval_rate_threshold: float = 60.0,
    risk_score_threshold: float = 70.0,
    tier3_concentration_threshold: float = 40.0,
    exposure_threshold_lakhs: float = 5000.0,
) -> list[dict]:
    """Return portfolio risk alerts computed from persisted underwriting decisions."""
    analytics = get_portfolio_analytics_v2(mode)

    total_decisions = int(analytics.get("total_decisions", 0))
    approval_rate = round(float(analytics.get("approval_rate", 0.0)), 2)
    average_risk_score = round(float(analytics.get("average_risk_score", 0.0)), 2)
    total_exposure_lakhs = round(float(analytics.get("total_exposure_lakhs", 0.0)), 2)

    tier_distribution = analytics.get("tier_distribution", {})
    tier3_count = int(tier_distribution.get("Tier 3", 0)) if isinstance(tier_distribution, dict) else 0
    tier3_concentration = round((tier3_count / total_decisions) * 100, 2) if total_decisions else 0.0

    alerts: list[dict] = []

    if total_decisions > 0 and approval_rate < approval_rate_threshold:
        alerts.append(
            {
                "type": "APPROVAL_RATE_DROP",
                "severity": "HIGH",
                "message": "Approval rate dropped below configured threshold.",
                "value": approval_rate,
                "threshold": round(approval_rate_threshold, 2),
            }
        )

    if total_decisions > 0 and average_risk_score > risk_score_threshold:
        alerts.append(
            {
                "type": "AVERAGE_RISK_SCORE_HIGH",
                "severity": "MEDIUM",
                "message": "Average portfolio risk score exceeded configured threshold.",
                "value": average_risk_score,
                "threshold": round(risk_score_threshold, 2),
            }
        )

    if total_decisions > 0 and tier3_concentration > tier3_concentration_threshold:
        alerts.append(
            {
                "type": "TIER3_CONCENTRATION",
                "severity": "MEDIUM",
                "message": "Tier 3 concentration exceeded configured threshold.",
                "value": tier3_concentration,
                "threshold": round(tier3_concentration_threshold, 2),
            }
        )

    if total_exposure_lakhs > exposure_threshold_lakhs:
        alerts.append(
            {
                "type": "EXPOSURE_LIMIT_BREACH",
                "severity": "HIGH",
                "message": "Total approved exposure breached configured threshold.",
                "value": total_exposure_lakhs,
                "threshold": round(exposure_threshold_lakhs, 2),
            }
        )

    return alerts
