"""PayGuard AI — fraud-explainer payment agent (Week 1 skeleton).

FastAPI backend: risk-scores PayPal-shaped transactions with the retrained
XGBoost model, exposes the PayPal sandbox Orders API client, and listens for
PayPal webhooks. No credentials are needed to import or run the scoring
endpoints; the PayPal client raises a clear error only when actually used
without credentials configured.
"""
