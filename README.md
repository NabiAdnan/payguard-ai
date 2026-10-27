# PayGuard AI — fraud detection that explains itself

A checkout that doesn't just block suspicious payments — it *explains* them.
PayPal sandbox checkout → transaction features → XGBoost risk score → (Week 2)
LLM plain-language explanation + approve / review / block recommendation →
React dashboard where a merchant sees *why* each payment was flagged.

Built for the [PayPal AI Hackathon](https://www.devpost.com/) (solo entry).
Deadline: **November 12, 2026** — submission planned for Nov 10–11.

## Architecture (Week 1)

```
[React Dashboard — Week 2] ──REST──▶ [FastAPI Backend] ──▶ [PayPal Sandbox APIs]
        ▲                                  │                        │
        │                                  ▼                        ▼
        │                           [Feature Builder]        Orders API (create/capture)
        │                                  │                 Webhooks (payment events)
        │                                  ▼
        │                           [XGBoost Risk Model]
        │                                  │
        └──── decision + explanation ── [LLM Explainer — Week 2]
```

| Component | Status | Notes |
|---|---|---|
| PayPal integration | ✅ Code-complete, not yet run live | Orders API create/capture + webhook listener; needs sandbox credentials (see Setup) |
| Feature builder | ✅ Done | Maps PayPal captures onto the model schema; simulated signals labelled (see below) |
| XGBoost risk model | ✅ Done | Retrained from the original gateway repo's dataset (see Model provenance) |
| LLM explainer | ⏳ Week 2 | Templated interface stubbed by config messages; LLM wires in behind it |
| Decision engine | ⏳ Week 2 | Config thresholds (Approve < 40, Review < 70, else Reject) already applied |
| React dashboard | ⏳ Week 2 | |
| Storage | — | In-memory for the demo; no database in Week 1 |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py        # builds model/ artifacts (one-time)
uvicorn app.main:app --reload
```

Then score a sample PayPal-shaped payload:

```bash
curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
  -d @tests/sample_payload_clean.json
# {"transaction_id":"TXN-...","prediction":0,"fraud_probability":0.0371,
#  "risk_score":3.71,"decision":"Approve","confidence":"Low",
#  "reason":"Low fraud risk", ...}
```

Run the smoke tests (no network, no credentials needed):

```bash
pip install pytest
python -m pytest tests/ -v
```

## PayPal sandbox setup (live round-trip — needs Adnan's credentials)

1. developer.paypal.com → Dashboard → Apps & Credentials → **Sandbox** → app
   **PayGuard AI** (Merchant app) → note the Client ID + Secret.
2. Copy `.env.example` to `.env` and fill in `PAYPAL_CLIENT_ID` /
   `PAYPAL_CLIENT_SECRET`. **Never commit `.env`** — it is gitignored.
3. The client (`app/paypal_client.py`) then supports order create → buyer
   approve → capture, and the webhook listener scores `PAYMENT.CAPTURE.COMPLETED`
   events. Until credentials exist, live calls return HTTP 503 with a clear
   message; scoring endpoints work without them.

## Model provenance — read this before quoting metrics

The risk model is a **retrained** XGBoost classifier, not the original
`.pkl`. The original fraud-gateway repo never committed its trained artifacts
(`backend/model/*.pkl` is gitignored). `scripts/train_model.py` replicates
that repo's `notebooks/03_Production_Pipeline.ipynb` exactly — same dataset
(`fraud_engineered.csv`, 100k rows), same column drops, same 80/20 stratified
split (seed 42), same frequency encoding, same `SMOTE(random_state=42)`, same
`XGBClassifier` hyperparameters (300 trees, depth 8, lr 0.05) — and saves
text-only artifacts (booster JSON + encoding manifest, no pickle).

Holdout results of **this** retrained model (20,000 rows) are in
`model/metrics.json`: **accuracy 99.8%, precision 92.6%, recall 73.3%,
ROC-AUC 0.9977** — close to the original model's reported figures, but these
are the numbers that may be quoted for PayGuard AI.

Two deliberate fixes vs. the notebook, documented in the training script:
frequency maps are computed from raw training values (the notebook recomputed
them after encoding, keying them by numeric frequencies), and the
`fillna(0)` is a real assignment (the notebook's chained-assignment fill did
not take effect under pandas copy-on-write).

## Honest signals: what PayPal gives the model, and what it doesn't

A PayPal capture carries amount, currency, payer country, and time. The
model's 35 features include many signals no PayPal payload contains (customer
age, job, city population, purchase history, distance, velocity). The feature
builder fills those with documented defaults derived from the training data
(medians, first-transaction assumptions) and **every `/score` response lists
`signals.real_paypal_signals` vs `signals.defaulted_signals`**, so a score is
never mistaken for evidence it doesn't have. The live round-trip (build-plan
step 2) will re-check this mapping against a real capture JSON and record
which fields survive contact with the real API.

## API

| Endpoint | Description |
|---|---|
| `GET /` | Service info |
| `GET /health` | Liveness + model status |
| `GET /model-info` | Retrained model metrics + provenance |
| `POST /score` | Score a PayPal-shaped payload (see `tests/sample_payload_*.json`) |
| `POST /webhooks/paypal` | PayPal webhook listener; capture events are scored. Signature verification is not yet enabled (no webhook ID/credentials) — events are reported with `signature_verified: false`. |
| `GET /paypal/order/{id}` | Live order lookup (503 until credentials are configured) |

## License

MIT — see [LICENSE](LICENSE).
