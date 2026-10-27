"""Configuration: risk thresholds (lifted from the original gateway repo) and
PayPal settings read from the environment. Importing this module never
requires credentials to exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

# Risk thresholds / labels, lifted as-is from the original gateway repo's
# backend/config/config.json. Scores are 0-100 (fraud probability percent).
_DEFAULT_CONFIG = {
    "risk_thresholds": {"approve": 40, "review": 70, "high": 90},
    "decisions": {"approve": "Approve", "review": "Review", "reject": "Reject"},
    "messages": {
        "approve": "Low fraud risk",
        "review": "Medium fraud risk - Manual verification required",
        "reject": "High fraud risk - Transaction rejected",
    },
    "confidence": {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "very_high": "Very High",
    },
}


def load_risk_config() -> dict:
    config_path = MODEL_DIR / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return _DEFAULT_CONFIG


RISK_CONFIG = load_risk_config()


class PayPalSettings:
    """PayPal REST settings. All values come from the environment; nothing is
    required at import time. ``is_configured`` gates live API calls."""

    def __init__(self) -> None:
        self.client_id = os.environ.get("PAYPAL_CLIENT_ID", "")
        self.client_secret = os.environ.get("PAYPAL_CLIENT_SECRET", "")
        self.env = os.environ.get("PAYPAL_ENV", "sandbox").lower()

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    @property
    def base_url(self) -> str:
        if self.env == "live":
            return "https://api-m.paypal.com"
        return "https://api-m.sandbox.paypal.com"


def get_paypal_settings() -> PayPalSettings:
    return PayPalSettings()
