"""Week 2 tests: explainer, decision engine, decision log, webhook persistence.

Run:  python -m pytest tests/ -v   (from the repo root, in .venv)
No network, no PayPal credentials, no LLM keys needed — the explainer
tests exercise the template path with keys explicitly removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import store
from app.decisions import decide, recommendation_for
from app.explainer import explain, explainer_config
from app.main import app
from app.schemas import PayPalTransaction

client = TestClient(app)
TESTS_DIR = Path(__file__).parent


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Point the decision log at a throwaway SQLite file per test."""
    monkeypatch.setenv("PAYGUARD_DB_PATH", str(tmp_path / "test.db"))
    yield
    monkeypatch.delenv("PAYGUARD_DB_PATH", raising=False)


@pytest.fixture()
def no_llm_keys(monkeypatch):
    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    yield


def _payload() -> PayPalTransaction:
    return PayPalTransaction(
        amount={"currency_code": "USD", "value": "49.99"},
        payer_country_code="US",
        payer_email="buyer@example.com",
        transaction_time="2026-10-27T15:30:00Z",
    )


# --- Explainer (template path) -------------------------------------------

def test_explainer_template_names_amount_country_and_top_feature(no_llm_keys):
    contributions = [
        {"feature": "amt", "value": "49.99", "contribution": -0.42},
        {"feature": "Is_Night", "value": "0", "contribution": -0.10},
    ]
    result = explain(_payload(), 3.71, "Approve", contributions, {"real_paypal_signals": ["amount"], "defaulted_signals": ["job"]})
    assert result["source"] == "template"
    assert "49.99" in result["explanation"]
    assert "US" in result["explanation"]
    assert "amt" in result["explanation"] or "payment amount" in result["explanation"]
    assert result["recommendation"] == "approve"


def test_explainer_recommendation_matches_decision(no_llm_keys):
    payload = _payload()
    assert explain(payload, 10.0, "Approve", [], {})["recommendation"] == "approve"
    assert explain(payload, 50.0, "Review", [], {})["recommendation"] == "review"
    assert explain(payload, 95.0, "Reject", [], {})["recommendation"] == "block"


def test_explainer_config_is_template_without_keys(no_llm_keys):
    assert explainer_config() == {"mode": "template", "provider": None, "model": None}


# --- Decision engine: thresholds verbatim from the lifted config ---------

@pytest.mark.parametrize(
    ("score", "decision", "confidence"),
    [
        (0.0, "Approve", "Low"),
        (39.99, "Approve", "Low"),
        (40.0, "Review", "Medium"),
        (69.99, "Review", "Medium"),
        (70.0, "Reject", "High"),
        (89.99, "Reject", "High"),
        (90.0, "Reject", "Very High"),
        (100.0, "Reject", "Very High"),
    ],
)
def test_decision_threshold_boundaries(score, decision, confidence):
    decided = decide(score)
    assert decided.decision == decision
    assert decided.confidence == confidence


def test_recommendation_for_labels():
    assert recommendation_for("Approve") == "approve"
    assert recommendation_for("Review") == "review"
    assert recommendation_for("Reject") == "block"


# --- Decision log endpoints ----------------------------------------------

def test_score_creates_log_entry(isolated_db):
    payload = json.loads((TESTS_DIR / "sample_payload_clean.json").read_text())
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"]
    assert body["explanation_source"] == "template"
    assert len(body["top_features"]) == 3
    assert body["feature_contributions"]
    assert body["contribution_method"] == "xgb_pred_contribs"

    listed = client.get("/decisions").json()
    assert len(listed) == 1
    assert listed[0]["id"] == body["transaction_id"]
    assert listed[0]["source"] == "api"

    fetched = client.get(f"/decisions/{body['transaction_id']}").json()
    assert fetched["explanation"] == body["explanation"]
    assert fetched["risk_score"] == body["risk_score"]


def test_decisions_list_newest_first_and_limit(isolated_db):
    for i in range(3):
        store.log_decision(
            {
                "id": f"TXN-TEST{i}",
                "amount": "10.00",
                "currency": "USD",
                "risk_score": 5.0,
                "decision": "Approve",
                "explanation": "test",
                "explanation_source": "template",
                "recommendation": "approve",
                "signals": {},
                "top_features": [],
                "source": "api",
            }
        )
    listed = client.get("/decisions?limit=2").json()
    assert len(listed) == 2
    assert listed[0]["id"] == "TXN-TEST2"


def test_get_unknown_decision_is_404(isolated_db):
    assert client.get("/decisions/TXN-NOPE").status_code == 404


def _review_record(record_id: str = "TXN-REVIEW1") -> None:
    store.log_decision(
        {
            "id": record_id,
            "amount": "250.00",
            "currency": "USD",
            "payer_country": "US",
            "risk_score": 55.0,
            "decision": "Review",
            "explanation": "Mixed signals.",
            "explanation_source": "template",
            "recommendation": "review",
            "signals": {"real_paypal_signals": ["amount"], "defaulted_signals": []},
            "top_features": [],
            "source": "api",
        }
    )


def test_review_action_approve(isolated_db):
    _review_record()
    response = client.post("/decisions/TXN-REVIEW1/action", json={"action": "approve"})
    assert response.status_code == 200
    assert response.json()["merchant_action"] == "approve"
    assert response.json()["action_at"] is not None


def test_review_action_block(isolated_db):
    _review_record("TXN-REVIEW2")
    response = client.post("/decisions/TXN-REVIEW2/action", json={"action": "block"})
    assert response.status_code == 200
    assert response.json()["merchant_action"] == "block"


def test_action_on_approve_decision_is_409(isolated_db):
    store.log_decision(
        {
            "id": "TXN-APPROVED",
            "amount": "10.00",
            "currency": "USD",
            "risk_score": 2.0,
            "decision": "Approve",
            "explanation": "Clean.",
            "explanation_source": "template",
            "recommendation": "approve",
            "signals": {},
            "top_features": [],
            "source": "api",
        }
    )
    response = client.post("/decisions/TXN-APPROVED/action", json={"action": "block"})
    assert response.status_code == 409


def test_second_action_on_same_review_is_409(isolated_db):
    _review_record()
    assert client.post("/decisions/TXN-REVIEW1/action", json={"action": "approve"}).status_code == 200
    assert client.post("/decisions/TXN-REVIEW1/action", json={"action": "block"}).status_code == 409


def test_action_on_unknown_decision_is_404(isolated_db):
    response = client.post("/decisions/TXN-NOPE/action", json={"action": "approve"})
    assert response.status_code == 404


def test_invalid_action_value_is_422(isolated_db):
    _review_record()
    response = client.post("/decisions/TXN-REVIEW1/action", json={"action": "refund"})
    assert response.status_code == 422


# --- Webhook persistence ---------------------------------------------------

def test_webhook_capture_event_lands_in_log(isolated_db):
    event = {
        "event_type": "PAYMENT.CAPTURE.COMPLETED",
        "create_time": "2026-10-27T18:06:52Z",
        "resource": {
            "amount": {"currency_code": "USD", "value": "49.99"},
            "payer": {
                "email_address": "sb-wsjba53580789@personal.example.com",
                "address": {"country_code": "US"},
            },
            "shipping": {
                "address": {
                    "city": "San Jose",
                    "admin_area_1": "CA",
                    "country_code": "US",
                }
            },
        },
    }
    response = client.post("/webhooks/paypal", json=event)
    assert response.status_code == 200
    body = response.json()
    assert body["signature_verified"] is False
    assert body["transaction_id"] is not None
    assert body["explanation"]

    listed = client.get("/decisions").json()
    assert len(listed) == 1
    assert listed[0]["id"] == body["transaction_id"]
    assert listed[0]["source"] == "webhook"
    assert listed[0]["payer_email"] == "sb-wsjba53580789@personal.example.com"
    assert "city" in listed[0]["signals"]["real_paypal_signals"]
