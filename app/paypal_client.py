"""PayPal sandbox REST client (Orders API v2).

Complete but NOT run live in Week 1: no credentials have been shared yet.
Credentials come from the environment (see .env.example) and are read lazily
— importing this module and constructing the client never touches the
network. Every method raises PayPalCredentialsMissing with a clear message
when credentials are absent, instead of failing obscurely.
"""

from __future__ import annotations

import time

import httpx

from .config import PayPalSettings


class PayPalCredentialsMissing(RuntimeError):
    """Raised when a live PayPal call is attempted without credentials."""


class PayPalClient:
    def __init__(self, settings: PayPalSettings | None = None) -> None:
        from .config import get_paypal_settings

        self.settings = settings or get_paypal_settings()
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def _require_credentials(self) -> None:
        if not self.settings.is_configured:
            raise PayPalCredentialsMissing(
                "PayPal credentials are not configured. Set PAYPAL_CLIENT_ID and "
                "PAYPAL_CLIENT_SECRET in .env (from developer.paypal.com → "
                "Dashboard → Apps & Credentials → Sandbox)."
            )

    def get_access_token(self) -> str:
        """OAuth 2.0 client-credentials grant. Cached until expiry."""
        self._require_credentials()
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        response = httpx.post(
            f"{self.settings.base_url}/v1/oauth2/token",
            auth=(self.settings.client_id, self.settings.client_secret),
            data={"grant_type": "client_credentials"},
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._token_expires_at = time.time() + int(data.get("expires_in", 32400))
        return self._token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_access_token()}",
            "Content-Type": "application/json",
        }

    def create_order(self, amount_value: str, currency_code: str = "USD") -> dict:
        """Create a PayPal order (intent CAPTURE). Returns the order JSON,
        including the buyer-approval link."""
        response = httpx.post(
            f"{self.settings.base_url}/v2/checkout/orders",
            headers=self._headers(),
            json={
                "intent": "CAPTURE",
                "purchase_units": [
                    {"amount": {"currency_code": currency_code, "value": amount_value}}
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def capture_order(self, order_id: str) -> dict:
        """Capture an approved order. Returns the full capture JSON — saved
        verbatim, it becomes the contract the feature builder is written
        against (see the build plan, step 2)."""
        response = httpx.post(
            f"{self.settings.base_url}/v2/checkout/orders/{order_id}/capture",
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_order(self, order_id: str) -> dict:
        response = httpx.get(
            f"{self.settings.base_url}/v2/checkout/orders/{order_id}",
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
