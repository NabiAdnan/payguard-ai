"""PayGuard AI FastAPI application — Week 1 skeleton.

Endpoints:
  GET  /                  service info
  GET  /health            liveness + model status
  GET  /model-info        retrained model metrics + provenance
  POST /score             score a PayPal-shaped transaction payload
  POST /webhooks/paypal   PayPal webhook listener (capture events auto-scored)

The webhook listener is complete code, but signature verification needs the
sandbox credentials + webhook ID, which have not been shared yet. Until then
events are accepted, scored, and reported with signature_verified=false —
clearly labelled, never silently trusted. (Verification is a Week 2
hardening item, before any public deployment.)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from . import model as risk_model
from .config import RISK_CONFIG
from .features import build_features
from .paypal_client import PayPalClient, PayPalCredentialsMissing
from .schemas import PayPalTransaction, ScoreResponse, WebhookResponse


def _decision(score: float) -> tuple[str, str, str]:
    """Decision / confidence / reason from the lifted risk config (0-100 scale)."""
    thresholds = RISK_CONFIG["risk_thresholds"]
    if score < thresholds["approve"]:
        return (
            RISK_CONFIG["decisions"]["approve"],
            RISK_CONFIG["confidence"]["low"],
            RISK_CONFIG["messages"]["approve"],
        )
    if score < thresholds["review"]:
        return (
            RISK_CONFIG["decisions"]["review"],
            RISK_CONFIG["confidence"]["medium"],
            RISK_CONFIG["messages"]["review"],
        )
    return (
        RISK_CONFIG["decisions"]["reject"],
        RISK_CONFIG["confidence"]["high"]
        if score < thresholds["high"]
        else RISK_CONFIG["confidence"]["very_high"],
        RISK_CONFIG["messages"]["reject"],
    )


def _score_features(features: dict) -> dict:
    result = risk_model.score_transaction(features)
    risk_score = round(result["fraud_probability"] * 100, 2)
    decision, confidence, reason = _decision(risk_score)
    return {
        "transaction_id": f"TXN-{str(uuid.uuid4())[:8].upper()}",
        "prediction": result["prediction"],
        "fraud_probability": round(result["fraud_probability"], 4),
        "risk_score": risk_score,
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
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
        "XGBoost risk scoring. Week 1 skeleton — scoring + PayPal integration "
        "code; LLM explainer and dashboard land in Week 2."
    ),
    version="0.1.0",
    lifespan=lifespan,
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
    }


@app.post("/score", response_model=ScoreResponse)
def score(payload: PayPalTransaction) -> ScoreResponse:
    if not risk_model.is_loaded():
        raise HTTPException(
            status_code=503, detail=f"Model unavailable: {risk_model.load_error()}"
        )
    features, provenance = build_features(payload)
    scored = _score_features(features)
    return ScoreResponse(
        **scored,
        paypal={
            "amount": payload.amount.value,
            "currency_code": payload.amount.currency_code,
            "payer_country_code": payload.payer_country_code,
        },
        signals=provenance,
        top_features=[],
    )


@app.post("/webhooks/paypal", response_model=WebhookResponse)
async def paypal_webhook(request: Request) -> WebhookResponse:
    """PayPal webhook listener. Handles PAYMENT.CAPTURE.COMPLETED (and
    CHECKOUT.ORDER.APPROVED as an acknowledgement); other events are acked
    without scoring. Signature verification is not yet enabled (no webhook ID
    / credentials shared) — see module docstring."""
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
    payload = PayPalTransaction(
        amount={
            "currency_code": amount.get("currency_code", "USD"),
            "value": str(amount.get("value", "0")),
        },
        payer_country_code=(payer.get("address") or {}).get("country_code"),
        payer_email=payer.get("email_address"),
        transaction_time=event.get("create_time"),
    )
    features, _ = build_features(payload)
    scored = _score_features(features)
    return WebhookResponse(
        received=True,
        event_type=event_type,
        risk_score=scored["risk_score"],
        decision=scored["decision"],
        signature_verified=False,
        detail="Scored without signature verification (credentials not configured).",
    )


@app.get("/paypal/order/{order_id}")
def paypal_order(order_id: str) -> dict:
    """Live PayPal lookup (requires sandbox credentials). Week 1: code-complete,
    not run — credentials arrive via the secure route."""
    try:
        return PayPalClient().get_order(order_id)
    except PayPalCredentialsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
