"""Explainer: risk score + top features -> plain-language explanation.

One interface, two backends behind it:

  * Template (always available). A deterministic 2-3 sentence explanation
    built from the actual transaction: amount, payer country, time, and the
    top contributing features. The response labels its source "template" —
    it never pretends an LLM wrote it.
  * Real LLM (optional, environment-configured only). Gemini free tier via
    ``GOOGLE_API_KEY``/``GEMINI_API_KEY``, or OpenAI via ``OPENAI_API_KEY``.
    Provider/model are chosen with ``LLM_PROVIDER`` / ``LLM_MODEL`` /
    ``GEMINI_MODEL`` / ``OPENAI_MODEL``. No key is ever requested, invented,
    logged, or stored by this code — keys live only in the environment.
    If no key is configured, or the provider call fails for any reason, the
    template path runs instead. LLM results are cached per transaction.

The dashboard and decision engine only ever see
``explain(...) -> {"explanation", "recommendation", "source"}``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone

import httpx

from .contributions import display_value
from .decisions import recommendation_for
from .schemas import PayPalTransaction

# Plain-language glosses for the model's raw features, used by the template
# path so explanations name what a feature means, not just its column name.
_GLOSSES = {
    "amt": "the payment amount",
    "Log_Amount": "the payment amount (log scale)",
    "High_Amount_Flag": "an unusually large amount",
    "Transaction_Hour": "the time of day",
    "Is_Night": "a night-time transaction",
    "Is_Weekend": "a weekend transaction",
    "Day_of_Week": "the day of the week",
    "Distance_km": "the distance from the customer's usual location",
    "Customer_Avg_Distance": "the customer's usual transaction distance",
    "Distance_Ratio": "the distance relative to the customer's usual range",
    "Customer_Avg_Spend": "the customer's average spend",
    "Spend_Ratio": "the spend relative to the customer's average",
    "Amount_Difference": "the change from the previous transaction amount",
    "Amount_Growth_Ratio": "the growth over the previous transaction amount",
    "Previous_Transaction_Amount": "the previous transaction amount",
    "Time_Since_Last_Transaction": "the time since the last transaction",
    "Customer_Transaction_Count": "the customer's transaction history",
    "Is_First_Transaction": "a first-ever transaction for this customer",
    "Merchant_Popularity": "how often this merchant is used",
    "Merchant_Visited_Count": "how often this merchant has been visited",
    "Is_Favorite_Merchant": "whether this is a favourite merchant",
    "Is_Favorite_Category": "whether this is a favourite category",
    "city_pop": "the population of the transaction city",
    "Customer_Age": "the customer's age",
    "job": "the customer's occupation",
    "category": "the purchase category",
    "Merchant_Category": "the merchant category",
    "Transaction_Type": "the transaction type",
    "Payment_Method": "the payment method",
    "gender": "the customer's gender",
    "city": "the transaction city",
    "state": "the transaction state",
    "merchant": "the merchant",
    "Amount_Zscore": "how far the amount is from the customer's usual spend",
    "Spend_Percentile": "where the amount sits in the customer's spend range",
}

_CACHE_MAX = 512
_cache: OrderedDict[str, dict] = OrderedDict()

# Binary flags read backwards when their value is 0 ("an unusually large
# amount" is wrong for High_Amount_Flag = 0). Zero-valued readings:
_ZERO_GLOSSES = {
    "High_Amount_Flag": "no unusually large amount",
    "Is_Night": "a daytime transaction",
    "Is_Weekend": "a weekday transaction",
    "Is_First_Transaction": "a returning customer, not a first transaction",
    "Is_Favorite_Merchant": "not a favourite merchant",
    "Is_Favorite_Category": "not a favourite category",
}


def _driver_gloss(feature: str, value: str) -> str:
    try:
        is_zero = float(value) == 0.0
    except (TypeError, ValueError):
        is_zero = False
    if is_zero and feature in _ZERO_GLOSSES:
        return _ZERO_GLOSSES[feature]
    return _GLOSSES.get(feature, feature)


def explainer_config() -> dict:
    """Report the explainer mode without ever exposing key material."""
    provider = _llm_provider()
    if provider is None:
        return {"mode": "template", "provider": None, "model": None}
    return {"mode": "llm", "provider": provider[0], "model": provider[1]}


def _llm_provider() -> tuple[str, str] | None:
    """(provider, model) if an LLM key is configured, else None."""
    requested = os.environ.get("LLM_PROVIDER", "").strip().lower()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    gemini_key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if requested == "openai" and openai_key:
        return ("openai", os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini")
    if requested == "gemini" and gemini_key:
        return ("gemini", os.environ.get("GEMINI_MODEL") or os.environ.get("LLM_MODEL") or "gemini-1.5-flash")
    if requested:
        return None  # Requested provider has no key: template path, honestly.
    if openai_key:
        return ("openai", os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini")
    if gemini_key:
        return ("gemini", os.environ.get("GEMINI_MODEL") or os.environ.get("LLM_MODEL") or "gemini-1.5-flash")
    return None


def _cache_key(payload: PayPalTransaction, risk_score: float, contributions: list[dict]) -> str:
    canonical = json.dumps(
        {
            "amount": payload.amount.value,
            "currency": payload.amount.currency_code,
            "country": payload.payer_country_code,
            "email": payload.payer_email,
            "time": payload.transaction_time,
            "risk_score": round(risk_score, 2),
            "contributions": contributions,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _format_time(transaction_time: str | None) -> str:
    if not transaction_time:
        return ""
    try:
        moment = datetime.fromisoformat(transaction_time.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return f" at {moment.strftime('%H:%M')} UTC"
    except ValueError:
        return ""


def _risk_band(risk_score: float) -> str:
    if risk_score < 40:
        return "low risk"
    if risk_score < 70:
        return "medium risk"
    return "high risk"


def _template_explain(
    payload: PayPalTransaction,
    risk_score: float,
    decision: str,
    contributions: list[dict],
) -> dict:
    country = payload.payer_country_code or "an unknown country"
    time_phrase = _format_time(payload.transaction_time)
    summary = (
        f"This {payload.amount.value} {payload.amount.currency_code} payment from a "
        f"{country} payer{time_phrase} scored {risk_score:.2f} out of 100, "
        f"which is {_risk_band(risk_score)}."
    )
    if contributions:
        drivers = []
        for contrib in contributions:
            gloss = _driver_gloss(contrib["feature"], contrib["value"])
            direction = "pushed the risk score up" if contrib["contribution"] > 0 else "pushed the risk score down"
            drivers.append(f"{gloss} ({contrib['feature']} = {display_value(contrib['value'])}) {direction}")
        drivers_sentence = "The strongest signals were " + "; ".join(drivers) + "."
    else:
        drivers_sentence = "No single feature dominated the score."
    action = {
        "Approve": "Nothing looks suspicious, so the payment can be approved as normal.",
        "Review": "The signals are mixed, so hold this payment for a quick manual check before capturing it.",
        "Reject": "The risk signals outweigh the payment, so this payment should be blocked.",
    }.get(decision, "Hold this payment for a manual check.")
    return {
        "explanation": f"{summary} {drivers_sentence} {action}",
        "recommendation": recommendation_for(decision),
        "source": "template",
    }


def _llm_prompt(
    payload: PayPalTransaction,
    risk_score: float,
    decision: str,
    contributions: list[dict],
    signals: dict,
) -> str:
    drivers = "\n".join(
        f"- {c['feature']} = {c['value']} (log-odds contribution {c['contribution']:+.2f})"
        for c in contributions
    )
    return (
        "You explain fraud risk scores to a merchant in plain language. "
        "Write a 2-3 sentence explanation of this payment's risk score. "
        "Mention the amount, payer country, and the top contributing features. "
        "Do not invent facts that are not in the data below; several model "
        "features were filled with documented defaults, so do not present "
        "them as observed evidence. Reply with JSON only: "
        '{"explanation": "...", "recommendation": "approve|review|block"}.\n\n'
        f"Amount: {payload.amount.value} {payload.amount.currency_code}\n"
        f"Payer country: {payload.payer_country_code or 'unknown'}\n"
        f"Payer email: {payload.payer_email or 'unknown'}\n"
        f"Transaction time (UTC): {payload.transaction_time or 'unknown'}\n"
        f"Risk score: {risk_score:.2f}/100, decision: {decision}\n"
        f"Top contributing features:\n{drivers or '(none)'}\n"
        f"Real PayPal signals: {', '.join(signals.get('real_paypal_signals', []))}\n"
        f"Defaulted signals: {', '.join(signals.get('defaulted_signals', []))}"
    )


def _parse_llm_json(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("explanation"), str):
        return None
    return parsed


def _call_gemini(prompt: str, model: str, api_key: str) -> dict | None:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    try:
        response = httpx.post(
            url,
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15.0,
        )
        if response.status_code != 200:
            return None
        candidates = response.json().get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None
    return _parse_llm_json(text)


def _call_openai(prompt: str, model: str, api_key: str) -> dict | None:
    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You explain fraud risk scores to merchants. Reply with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=15.0,
        )
        if response.status_code != 200:
            return None
        content = response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None
    return _parse_llm_json(content)


def explain(
    payload: PayPalTransaction,
    risk_score: float,
    decision: str,
    contributions: list[dict],
    signals: dict,
) -> dict:
    """Explain one scored transaction. Always returns an explanation.

    Source is "template" unless a configured LLM produced the text, in
    which case it is "llm:gemini" or "llm:openai". LLM failures fall back
    to the template path — the caller never sees an error for this.
    """
    template = _template_explain(payload, risk_score, decision, contributions)
    provider = _llm_provider()
    if provider is None:
        return template

    key = _cache_key(payload, risk_score, contributions)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    name, model = provider
    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        if name == "openai"
        else (
            os.environ.get("GOOGLE_API_KEY", "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
        )
    )
    prompt = _llm_prompt(payload, risk_score, decision, contributions, signals)
    parsed = _call_gemini(prompt, model, api_key) if name == "gemini" else _call_openai(prompt, model, api_key)
    if parsed is None:
        return template
    recommendation = parsed.get("recommendation")
    if recommendation not in {"approve", "review", "block"}:
        recommendation = recommendation_for(decision)
    result = {
        "explanation": parsed["explanation"],
        "recommendation": recommendation,
        "source": f"llm:{name}",
    }
    _cache[key] = result
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return result
