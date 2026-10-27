"""Decision log: every scored transaction, persisted in SQLite.

Stdlib ``sqlite3`` only. The database file lives at ``data/payguard.db``
(gitignored) unless ``PAYGUARD_DB_PATH`` is set — tests point it at a
temporary file. The schema is created lazily on first use, so importing
this module never touches the filesystem.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import BASE_DIR

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    amount TEXT NOT NULL,
    currency TEXT NOT NULL,
    payer_country TEXT,
    payer_email TEXT,
    risk_score REAL NOT NULL,
    decision TEXT NOT NULL,
    explanation TEXT NOT NULL,
    explanation_source TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    top_features_json TEXT NOT NULL,
    source TEXT NOT NULL,
    merchant_action TEXT,
    action_at TEXT
)
"""


def get_db_path() -> Path:
    override = os.environ.get("PAYGUARD_DB_PATH", "").strip()
    if override:
        return Path(override)
    return BASE_DIR / "data" / "payguard.db"


def _connect() -> sqlite3.Connection:
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def log_decision(record: dict) -> None:
    """Persist one scored transaction. ``record`` keys match the columns."""
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO decisions
                (id, created_at, amount, currency, payer_country, payer_email,
                 risk_score, decision, explanation, explanation_source,
                 recommendation, signals_json, top_features_json, source,
                 merchant_action, action_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    record["id"],
                    record.get("created_at", datetime.now(timezone.utc).isoformat()),
                    record["amount"],
                    record["currency"],
                    record.get("payer_country"),
                    record.get("payer_email"),
                    record["risk_score"],
                    record["decision"],
                    record["explanation"],
                    record["explanation_source"],
                    record["recommendation"],
                    json.dumps(record.get("signals", {})),
                    json.dumps(record.get("top_features", [])),
                    record.get("source", "api"),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _row_to_record(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "amount": row["amount"],
        "currency": row["currency"],
        "payer_country": row["payer_country"],
        "payer_email": row["payer_email"],
        "risk_score": row["risk_score"],
        "decision": row["decision"],
        "explanation": row["explanation"],
        "explanation_source": row["explanation_source"],
        "recommendation": row["recommendation"],
        "signals": json.loads(row["signals_json"]),
        "top_features": json.loads(row["top_features_json"]),
        "source": row["source"],
        "merchant_action": row["merchant_action"],
        "action_at": row["action_at"],
    }


def list_decisions(limit: int = 50) -> list[dict]:
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        finally:
            conn.close()
    return [_row_to_record(r) for r in rows]


def get_decision(decision_id: str) -> dict | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
        finally:
            conn.close()
    return _row_to_record(row) if row is not None else None


def record_action(decision_id: str, action: str) -> str:
    """Record a merchant review-queue action.

    Returns "ok", "not_found", or "not_actionable". Only "Review" items
    with no action recorded yet can be actioned; anything else is a 409
    at the API layer — a merchant must not approve/block an already
    decided or already-actioned transaction through this endpoint.
    """
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT decision, merchant_action FROM decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
            if row is None:
                return "not_found"
            if row["decision"] != "Review" or row["merchant_action"] is not None:
                return "not_actionable"
            conn.execute(
                "UPDATE decisions SET merchant_action = ?, action_at = ? WHERE id = ?",
                (action, datetime.now(timezone.utc).isoformat(), decision_id),
            )
            conn.commit()
            return "ok"
        finally:
            conn.close()
