# PayGuard AI — fraud detection that explains itself

A checkout that doesn't just block suspicious payments — it *explains* them.
PayPal sandbox checkout → transaction features → XGBoost risk score → LLM
plain-language explanation + approve / review / block recommendation →
React dashboard where a merchant sees *why* each payment was flagged and can
act on the review queue.

Built for the [PayPal AI Hackathon](https://paypalaihackathon.devpost.com/)
(solo entry). Deadline: **November 12, 2026** — submission planned for
Nov 10–11.

## Status: Week 2 done (ahead of the Nov 6 milestone)

Week 1 (Oct 24–30) milestone — one real sandbox payment in, one risk score
out — was proven Oct 27: a $49.99 USD sandbox order was approved by a US
sandbox buyer account, captured (`COMPLETED`), and scored 1.36 → Approve.
Week 2 (Oct 31–Nov 6) is implemented: top contributing features, the
explainer (template path live; LLM behind the same interface), the decision
engine + SQLite decision log, webhook auto-scoring/persistence, and the
React dashboard. Remaining before submission (Week 3): scripted demo
scenarios, the < 3 min video, and the Devpost submission itself.

## Architecture

```
[React Dashboard] ──REST──▶ [FastAPI Backend] ──▶ [PayPal Sandbox APIs]
        ▲                          │                        │
        │                          ▼                        ▼
        │                   [Feature Builder]        Orders API (create/capture)
        │                          │                 Webhooks (payment events)
        │                          ▼
        │                   [XGBoost Risk Model]
        │                          │
        └──── decision + explanation ── [LLM Explainer]
                          │                │
                          ▼                ▼
                   [Decision Engine]  [SQLite Decision Log]
```

| Component | Status | Notes |
|---|---|---|
| PayPal integration | ✅ Live round-trip proven Oct 27 | Orders API create/capture + webhook listener; needs sandbox credentials (see Setup) |
| Feature builder | ✅ Done | Maps PayPal captures onto the model schema; simulated signals labelled (see below) |
| XGBoost risk model | ✅ Done | Retrained from the original gateway repo's dataset (see Model provenance) |
| Top contributing features | ✅ Done | XGBoost native per-prediction contributions (`pred_contribs`, tree-SHAP approximation), aggregated from one-hot blocks back to the 35 raw features; falls back to global gain importances if contributions are unavailable, and the method used is reported in each response |
| LLM explainer | ✅ Template live; LLM optional | `app/explainer.py`: one interface. Template explanations reference the real amount, payer country, time, and top features, and are labelled `source: "template"`. A real LLM (Gemini free tier / OpenAI) runs behind the same interface when a key is configured via the environment — see Setup |
| Decision engine | ✅ Done | `app/decisions.py`; thresholds from the config (Approve < 30, Review < 70, else Reject — approve cut lowered from the lifted 40 to 30 on Oct 28 so borderline scores go to human review) |
| Decision log | ✅ Done | SQLite at `data/payguard.db` (gitignored); every scored transaction logged with its explanation; review-queue actions via the API |
| React dashboard | ✅ Done | `dashboard/` (Vite + React + TypeScript): live feed, detail view, review-queue approve/block, stats header |
| Storage | ✅ SQLite | `data/payguard.db`; override with `PAYGUARD_DB_PATH` |

## Quickstart

Backend:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py        # builds model/ artifacts (one-time)
uvicorn app.main:app --reload        # http://127.0.0.1:8000
```

Then score a sample PayPal-shaped payload:

```bash
curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
  -d @tests/sample_payload_clean.json
# {"transaction_id":"TXN-...","prediction":0,"fraud_probability":0.0371,
#  "risk_score":3.71,"decision":"Approve","confidence":"Low",
#  "reason":"Low fraud risk","explanation":"This 49.99 USD payment ...",
#  "explanation_source":"template","recommendation":"approve", ...}
```

Dashboard (separate terminal, backend running on :8000):

```bash
cd dashboard
npm install
npm run dev                          # http://localhost:5173
```

Production build of the dashboard:

```bash
cd dashboard
npm run build                        # outputs dashboard/dist/
npm run preview                      # serves the production build
```

To point the dashboard at a backend on another host/port, build with
`VITE_API_URL` set, e.g. `VITE_API_URL=https://api.example.com npm run build`.
Default is `http://localhost:8000`.

Run the tests (no network, no credentials needed):

```bash
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

Two setup findings from the Oct 27 live run, recorded so nobody re-learns
them: (1) an India (IN) sandbox buyer account rejects USD orders
(`UNSUPPORTED_PAYER_CURRENCY`), and INR is not supported by PayPal's Orders
API at all (`CURRENCY_NOT_SUPPORTED`) — the working buyer is a US sandbox
Personal account; (2) the sandbox checkout page can loop back to payment
selection without showing confirmation even though the approval *did*
record server-side — always verify order status via the Orders API before
retrying.

## Explainer: template vs LLM

`POST /score` and webhook scoring always return an `explanation`,
`recommendation` (`approve` / `review` / `block`), and
`explanation_source`. Today, with no LLM key configured, the source is
`"template"`: deterministic text built from the transaction's real amount,
payer country, time, and top contributing features. It is never presented
as LLM output.

To enable the LLM path, set **one** of these in the environment (never in
code or in a committed file). Gemini (free tier) is the project's chosen
provider: when its key is present it is used by default.

```bash
GOOGLE_API_KEY=...        # or GEMINI_API_KEY — Gemini free tier (chosen provider)
OPENAI_API_KEY=...        # OpenAI
LLM_PROVIDER=gemini       # optional: force a provider when both keys exist
LLM_MODEL=...             # optional: override the default model
```

Defaults: `gemini-1.5-flash` for Gemini, `gpt-4o-mini` for OpenAI. LLM
results are cached per transaction (in memory). If the provider call fails
for any reason, the template path runs instead and the source stays
`"template"`. `GET /model-info` reports the current explainer mode
(`template` or `llm`) without exposing any key material.

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
never mistaken for evidence it doesn't have. The live round-trip (Oct 27)
re-checked this mapping against a real capture JSON: amount, currency, payer
country/email, shipping city/state, and capture time survive; the rest remain
defaults, as labelled.

Two more honesty notes:

- **Top features are XGBoost's own per-prediction contributions**
  (`pred_contribs`), aggregated back to the 35 raw feature names — not the
  `shap` package, which is not installed. The method used is reported as
  `contribution_method` in each response.
- **Webhook signature verification is pending credentials.** Capture events
  are accepted, scored, explained, and logged with
  `signature_verified: false`. No signature is ever faked; verification must
  be wired (webhook ID + credentials) before any public deployment.

## API

| Endpoint | Description |
|---|---|
| `GET /` | Service info |
| `GET /health` | Liveness + model status |
| `GET /model-info` | Retrained model metrics + provenance + explainer mode |
| `POST /score` | Score a PayPal-shaped payload (see `tests/sample_payload_*.json`); logs the decision with its explanation |
| `POST /webhooks/paypal` | PayPal webhook listener; capture events are scored, explained, and logged. Signature verification is not yet enabled (no webhook ID/credentials) — events are reported with `signature_verified: false`. |
| `GET /paypal/order/{id}` | Live order lookup (503 until credentials are configured) |
| `GET /decisions?limit=50` | Decision log, newest first |
| `GET /decisions/{id}` | One logged decision |
| `POST /decisions/{id}/action` | Merchant review-queue action: `{"action": "approve" \| "block"}`. Only un-actioned `Review` decisions; anything else is 409. |

## License

MIT — see [LICENSE](LICENSE).
