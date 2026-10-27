"""Smoke tests for the Week 1 skeleton.

Run:  python -m pytest tests/ -v
Requires the trained model artifacts (python scripts/train_model.py).
No network, no PayPal credentials needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
TESTS_DIR = Path(__file__).parent


def _payload(name: str) -> dict:
    return json.loads((TESTS_DIR / name).read_text())


def test_health_reports_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["status"] == "Healthy"


def test_model_info_reports_metrics():
    response = client.get("/model-info")
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert metrics["roc_auc"] > 0.9
    assert metrics["accuracy"] > 0.9


def test_score_clean_payload():
    response = client.post("/score", json=_payload("sample_payload_clean.json"))
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_score"] <= 100.0
    assert body["decision"] in {"Approve", "Review", "Reject"}
    assert body["paypal"]["amount"] == "49.99"
    assert "amount" in body["signals"]["real_paypal_signals"]


def test_score_risky_payload_scores_higher_than_clean():
    clean = client.post("/score", json=_payload("sample_payload_clean.json")).json()
    risky = client.post("/score", json=_payload("sample_payload_risky.json")).json()
    assert risky["risk_score"] > clean["risk_score"]


def test_score_rejects_missing_amount():
    response = client.post("/score", json={"payer_country_code": "US"})
    assert response.status_code == 422


def test_webhook_capture_event_is_scored_without_credentials():
    event = {
        "event_type": "PAYMENT.CAPTURE.COMPLETED",
        "create_time": "2026-10-27T15:30:00Z",
        "resource": {
            "amount": {"currency_code": "USD", "value": "120.00"},
            "payer": {
                "email_address": "buyer@example.com",
                "address": {"country_code": "US"},
            },
        },
    }
    response = client.post("/webhooks/paypal", json=event)
    assert response.status_code == 200
    body = response.json()
    assert body["received"] is True
    assert body["signature_verified"] is False
    assert body["risk_score"] is not None


def test_webhook_unknown_event_acknowledged():
    response = client.post(
        "/webhooks/paypal", json={"event_type": "BILLING.SUBSCRIPTION.CREATED"}
    )
    assert response.status_code == 200
    assert response.json()["received"] is True


def test_paypal_order_lookup_without_credentials_is_503():
    response = client.get("/paypal/order/ORDER123")
    assert response.status_code == 503
    assert "PAYPAL_CLIENT_ID" in response.json()["detail"]
