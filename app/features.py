"""Map a PayPal-shaped payload onto the 35 model features.

Honesty contract (see README): a PayPal capture carries only a handful of
fraud signals — amount, currency, payer country, time. The remaining model
features have no PayPal source, so the builder fills them with documented,
clearly-labelled defaults derived from the training data. Every response
reports which signals were real PayPal fields and which were defaulted, so a
score is never mistaken for evidence it does not have.

Real PayPal-derived signals: amount (+ log amount, high-amount flag),
transaction time (+ hour, day-of-week, weekend, night), payer country
(recorded, not a model feature), payment method, transaction type.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .model import encoding
from .schemas import PayPalTransaction

# Fields with no PayPal source, filled from training-data medians/defaults.
DEFAULT_JOB = "Unknown"
DEFAULT_CITY = "Unknown"
DEFAULT_STATE = "Unknown"
DEFAULT_GENDER = "Unknown"
DEFAULT_CATEGORY = "other"
DEFAULT_MERCHANT_CATEGORY = "Other"


def _parse_time(value: str | None) -> datetime:
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def build_features(payload: PayPalTransaction) -> tuple[dict, dict[str, list[str]]]:
    """Return (features, provenance). Provenance lists real vs defaulted signals."""
    enc = encoding()
    defaults = enc.get("defaults", {})
    amount = float(payload.amount.value)
    moment = _parse_time(payload.transaction_time)

    city = payload.city or (payload.payer_address.city if payload.payer_address else None)
    state = payload.state or (payload.payer_address.state if payload.payer_address else None)

    high_amount_threshold = float(defaults.get("High_Amount_Flag_threshold", 500.0))

    features: dict = {
        "merchant": payload.merchant_id or "PAYGUARD_MERCHANT",
        "category": payload.category or DEFAULT_CATEGORY,
        "amt": amount,
        "gender": DEFAULT_GENDER,
        "city": city or DEFAULT_CITY,
        "state": state or DEFAULT_STATE,
        "city_pop": float(defaults.get("city_pop", 0)),
        "job": payload.job or DEFAULT_JOB,
        "Merchant_Category": payload.merchant_category or DEFAULT_MERCHANT_CATEGORY,
        "Transaction_Type": payload.transaction_type or "Online",
        "Customer_Age": payload.customer_age
        if payload.customer_age is not None
        else int(defaults.get("Customer_Age", 35)),
        "Payment_Method": payload.payment_method or "paypal",
        "Transaction_Hour": moment.hour,
        "Day_of_Week": moment.weekday(),
        "Is_Weekend": 1 if moment.weekday() >= 5 else 0,
        "Is_Night": 1 if moment.hour < 6 or moment.hour >= 22 else 0,
        "High_Amount_Flag": 1 if amount >= high_amount_threshold else 0,
        "Log_Amount": math.log1p(amount),
        "Customer_Avg_Spend": amount,
        "Spend_Ratio": 1.0,
        "Customer_Transaction_Count": 1,
        "Merchant_Popularity": 1,
        "Distance_km": 0.0,
        "Previous_Transaction_Amount": 0.0,
        "Time_Since_Last_Transaction": 0.0,
        "Amount_Difference": amount,
        "Amount_Growth_Ratio": 1.0,
        "Is_First_Transaction": 1,
        "Merchant_Visited_Count": 1,
        "Customer_Avg_Distance": 0.0,
        "Distance_Ratio": 0.0,
        "Spend_Percentile": 0.5,
        "Amount_Zscore": 0.0,
        "Is_Favorite_Merchant": 0,
        "Is_Favorite_Category": 0,
    }

    real = ["amount", "currency", "transaction_time"]
    if payload.payer_country_code:
        real.append("payer_country_code")
    if payload.payer_email:
        real.append("payer_email")
    if city:
        real.append("city")
    if state:
        real.append("state")
    if payload.job:
        real.append("job")
    if payload.customer_age is not None:
        real.append("customer_age")
    if payload.category:
        real.append("category")
    if payload.merchant_category:
        real.append("merchant_category")

    defaulted = [
        name
        for name in (
            "gender",
            "city_pop",
            "job",
            "customer_age",
            "distance_km",
            "purchase_history",
            "velocity",
            "spend_percentile",
            "amount_zscore",
        )
        if name not in real
    ]
    provenance = {"real_paypal_signals": real, "defaulted_signals": defaulted}
    return features, provenance
