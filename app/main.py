"""PayGuard AI FastAPI application.

Endpoints:
  GET  /                  service info
  GET  /health            liveness + model status
  GET  /model-info        retrained model metrics + provenance + explainer mode
  POST /score             score a PayPal-shaped transaction payload
  POST /webhooks/paypal   PayPal webhook listener (capture events auto-scored,
                          explained, and logged)
  GET  /paypal/order/{id} live PayPal lookup (needs sandbox credentials)
  GET  /decisions         decision log, newest first
  GET  /decisions/{id}    one logged decision
  POST /decisions/{id}/action  merchant review-queue action (approve/block)

The webhook listener accepts events and scores them, but signature
verification needs the sandbox credentials + webhook ID, which are not
configured in this environment. Until then events are accepted, scored,
and reported with signature_verified=false — clearly labelled, never
silently trusted. (Verification is a hardening item before any public
deployment.)

Explanations come from the template path unless an LLM key is configured
via the environment (see app/explainer.py). The response always labels
its explanation source, so a template explanation is never mistaken for
an LLM-written one.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import model as risk_model
from . import store
from .config import RISK_CONFIG
from .contributions import display_value, top_contributions
from .decisions import decide
from .explainer import explain, explainer_config
from .features import build_features
from .paypal_client import PayPalClient, PayPalCredentialsMissing
from .schemas import (
    ActionRequest,
    DecisionRecord,
    PayPalTransaction,
    ScoreResponse,
    WebhookResponse,
)


def _score_explain_and_log(
    payload: PayPalTransaction, source: str
) -> dict:
    """Score one payload, explain it, log the decision, return the record.

    Shared by POST /score and the webhook handler so both paths produce
    identical decisions, explanations, and log entries.
    """
    features, provenance = build_features(payload)
    result = risk_model.score_transaction(features)
    risk_score = round(result["fraud_probability"] * 100, 2)
    decision = decide(risk_score)
    contributions, method = top_contributions(features)
    explained = explain(
        payload, risk_score, decision.decision, contributions, provenance
    )
    top_features = [
        f"{c['feature']}={display_value(c['value'])} ({c['contribution']:+.2f} log-odds)"
        for c in contributions
    ]
    record = {
        "id": f"TXN-{str(uuid.uuid4())[:8].upper()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amount": payload.amount.value,
        "currency": payload.amount.currency_code,
        "payer_country": payload.payer_country_code,
        "payer_email": payload.payer_email,
        "risk_score": risk_score,
        "decision": decision.decision,
        "explanation": explained["explanation"],
        "explanation_source": explained["source"],
        "recommendation": explained["recommendation"],
        "signals": provenance,
        "top_features": top_features,
        "source": source,
    }
    try:
        store.log_decision(record)
    except Exception:  # noqa: BLE001 — scoring must survive a log failure
        # A failed write must not fail the score; the dashboard simply
        # won't show this transaction. Logged to stderr by the server.
        import sys

        print(f"payguard: failed to log decision {record['id']}", file=sys.stderr)
    return {
        "transaction_id": record["id"],
        "prediction": result["prediction"],
        "fraud_probability": round(result["fraud_probability"], 4),
        "risk_score": risk_score,
        "decision": decision.decision,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "feature_contributions": contributions,
        "contribution_method": method,
        "explanation": explained["explanation"],
        "explanation_source": explained["source"],
        "recommendation": explained["recommendation"],
        "top_features": top_features,
        "signals": provenance,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the model once at startup; /health reports the outcome.
    risk_model.is_loaded()
    yield


app = FastAPI(
    title="PayGuard AI",
    description=(
        "Fraud-explainer payment agent: PayPal sandbox checkout + "
        "XGBoost risk scoring + plain-language explanations + decision log."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

_cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def home() -> dict:
    return {"message": "Welcome to PayGuard AI", "service": "risk scoring"}


@app.get("/health")
def health() -> dict:
    status = {
        "status": "Healthy" if risk_model.is_loaded() else "Degraded",
        "model": "XGBoost",
        "model_loaded": risk_model.is_loaded(),
        "service": "Running",
    }
    if not risk_model.is_loaded():
        status["model_error"] = risk_model.load_error()
    return status


@app.get("/model-info")
def model_info() -> dict:
    metrics = risk_model.metrics()
    return {
        "model_name": "XGBoost Fraud Detection (retrained — see metrics provenance)",
        "metrics": metrics,
        "risk_thresholds": RISK_CONFIG["risk_thresholds"],
        "explainer": explainer_config(),
    }


@app.post("/score", response_model=ScoreResponse)
def score(payload: PayPalTransaction) -> ScoreResponse:
    if not risk_model.is_loaded():
        raise HTTPException(
            status_code=503, detail=f"Model unavailable: {risk_model.load_error()}"
        )
    scored = _score_explain_and_log(payload, source="api")
    return ScoreResponse(
        **scored,
        paypal={
            "amount": payload.amount.value,
            "currency_code": payload.amount.currency_code,
            "payer_country_code": payload.payer_country_code,
        },
    )


@app.post("/webhooks/paypal", response_model=WebhookResponse)
async def paypal_webhook(request: Request) -> WebhookResponse:
    """PayPal webhook listener. PAYMENT.CAPTURE.COMPLETED / DENIED events
    are scored, explained, and persisted to the decision log; other events
    are acked without scoring. Signature verification is not yet enabled
    (no webhook ID / credentials configured) — see module docstring."""
    event = await request.json()
    event_type = str(event.get("event_type", "unknown"))

    if event_type not in {"PAYMENT.CAPTURE.COMPLETED", "PAYMENT.CAPTURE.DENIED"}:
        return WebhookResponse(
            received=True,
            event_type=event_type,
            detail="Event acknowledged; only capture events are scored.",
        )

    resource = event.get("resource", {}) or {}
    amount = resource.get("amount", {}) or {}
    payer = resource.get("payer", {}) or {}
    payer_address = payer.get("address") or {}
    shipping_address = (resource.get("shipping") or {}).get("address") or {}
    payload = PayPalTransaction(
        amount={
            "currency_code": amount.get("currency_code", "USD"),
            "value": str(amount.get("value", "0")),
        },
        payer_country_code=payer_address.get("country_code"),
        payer_email=payer.get("email_address"),
        payer_address={
            "city": payer_address.get("city") or shipping_address.get("city"),
            "state": payer_address.get("state")
            or shipping_address.get("admin_area_1"),
            "country_code": payer_address.get("country_code"),
        },
        transaction_time=event.get("create_time"),
    )
    scored = _score_explain_and_log(payload, source="webhook")
    return WebhookResponse(
        received=True,
        event_type=event_type,
        risk_score=scored["risk_score"],
        decision=scored["decision"],
        signature_verified=False,
        detail="Scored without signature verification (credentials not configured).",
        transaction_id=scored["transaction_id"],
        explanation=scored["explanation"],
        recommendation=scored["recommendation"],
    )


@app.get("/decisions", response_model=list[DecisionRecord])
def list_decisions_endpoint(limit: int = 50) -> list[DecisionRecord]:
    return [DecisionRecord(**record) for record in store.list_decisions(limit)]


@app.get("/decisions/{decision_id}", response_model=DecisionRecord)
def get_decision_endpoint(decision_id: str) -> DecisionRecord:
    record = store.get_decision(decision_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return DecisionRecord(**record)


@app.post("/decisions/{decision_id}/action", response_model=DecisionRecord)
def record_action_endpoint(
    decision_id: str, action: ActionRequest
) -> DecisionRecord:
    """Merchant review-queue action. Only "Review" items with no action
    recorded yet can be actioned; anything else is a 409."""
    outcome = store.record_action(decision_id, action.action)
    if outcome == "not_found":
        raise HTTPException(status_code=404, detail="Decision not found")
    if outcome == "not_actionable":
        raise HTTPException(
            status_code=409,
            detail="Only un-actioned Review decisions can be actioned",
        )
    record = store.get_decision(decision_id)
    assert record is not None  # just actioned, must exist
    return DecisionRecord(**record)


@app.get("/paypal/order/{order_id}")
def paypal_order(order_id: str) -> dict:
    """Live PayPal lookup (requires sandbox credentials)."""
    try:
        return PayPalClient().get_order(order_id)
    except PayPalCredentialsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
