"""Request/response schemas for the PayGuard AI API.

The scoring input is a PayPal-shaped payload: the fields a PayPal Orders API
capture actually carries (amount, currency, payer country) plus optional
checkout context. Anything the model needs that PayPal does not provide is
filled by the feature builder with documented defaults — never silently.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PayPalAmount(BaseModel):
    currency_code: str = Field(..., examples=["USD"])
    value: str = Field(..., examples=["49.99"])


class PayerAddress(BaseModel):
    city: str | None = None
    state: str | None = None
    country_code: str | None = None


class PayPalTransaction(BaseModel):
    """Hand-written PayPal-shaped scoring payload (Week 1 contract)."""

    amount: PayPalAmount
    payer_country_code: str | None = Field(
        default=None, examples=["US"], description="Payer's country code, if known"
    )
    payer_email: str | None = None
    payer_address: PayerAddress | None = None
    merchant_id: str | None = Field(default=None, description="Our merchant identifier")
    merchant_category: str | None = None
    category: str | None = Field(
        default=None, description="Purchase category, e.g. grocery_pos"
    )
    payment_method: str | None = Field(default="paypal")
    transaction_type: str | None = Field(default="Online")
    transaction_time: str | None = Field(
        default=None,
        description="ISO-8601 timestamp of the capture; defaults to now (UTC)",
    )
    # Optional checkout context, when the caller knows it. These are NOT
    # PayPal API fields; they let demo scenarios express risk honestly.
    city: str | None = None
    state: str | None = None
    customer_age: int | None = None
    job: str | None = None


class ScoreResponse(BaseModel):
    transaction_id: str
    prediction: int
    fraud_probability: float
    risk_score: float
    decision: str
    confidence: str
    reason: str
    paypal: dict[str, Any]
    signals: dict[str, list[str]]
    top_features: list[str] = []


class WebhookResponse(BaseModel):
    received: bool
    event_type: str
    risk_score: float | None = None
    decision: str | None = None
    signature_verified: bool = False
    detail: str | None = None
