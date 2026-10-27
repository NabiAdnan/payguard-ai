import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchDecisions, recordAction, type Decision } from "./api";

function scoreClass(score: number): string {
  if (score < 40) return "low";
  if (score < 70) return "medium";
  return "high";
}

function formatTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

export default function App() {
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [acting, setActing] = useState(false);

  const load = useCallback(async () => {
    try {
      const rows = await fetchDecisions(50);
      setDecisions(rows);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach the backend.");
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), 5000);
    return () => clearInterval(timer);
  }, [load]);

  const selected =
    decisions.find((d) => d.id === selectedId) ?? decisions[0] ?? null;

  const stats = useMemo(() => {
    const total = decisions.length;
    const flagged = decisions.filter((d) => d.decision !== "Approve").length;
    const blockedAmount = decisions
      .filter((d) => d.decision === "Reject" || d.merchant_action === "block")
      .reduce((sum, d) => sum + Number(d.amount), 0);
    return {
      total,
      flagged,
      flaggedPct: total === 0 ? 0 : Math.round((flagged / total) * 100),
      blockedAmount,
    };
  }, [decisions]);

  async function act(action: "approve" | "block") {
    if (!selected || acting) return;
    setActing(true);
    setError(null);
    try {
      const updated = await recordAction(selected.id, action);
      setDecisions((rows) => rows.map((r) => (r.id === updated.id ? updated : r)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed.");
    } finally {
      setActing(false);
    }
  }

  const canAct =
    selected !== null &&
    selected.decision === "Review" &&
    selected.merchant_action === null;

  return (
    <div className="layout">
      <header>
        <h1>PayGuard AI</h1>
        <p className="subtitle">
          Fraud detection that explains itself — live transaction feed
        </p>
        <div className="stats">
          <div className="stat">
            <span className="stat-value">{stats.total}</span>
            <span className="stat-label">Transactions</span>
          </div>
          <div className="stat">
            <span className="stat-value">{stats.flaggedPct}%</span>
            <span className="stat-label">Flagged (review + reject)</span>
          </div>
          <div className="stat">
            <span className="stat-value">
              ${stats.blockedAmount.toFixed(2)}
            </span>
            <span className="stat-label">Blocked amount</span>
          </div>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <main>
        <section className="feed">
          <h2>Transactions</h2>
          {decisions.length === 0 && !error && (
            <p className="empty">
              No transactions yet. Score one via POST /score or send a
              PayPal webhook capture event — it will appear here.
            </p>
          )}
          <ul>
            {decisions.map((d) => (
              <li key={d.id}>
                <button
                  className={selected?.id === d.id ? "selected" : ""}
                  onClick={() => setSelectedId(d.id)}
                >
                  <span className={`badge ${scoreClass(d.risk_score)}`}>
                    {d.risk_score.toFixed(1)}
                  </span>
                  <span className="row-main">
                    <strong>
                      {d.amount} {d.currency}
                    </strong>{" "}
                    · {d.decision}
                    {d.merchant_action && ` · merchant: ${d.merchant_action}`}
                  </span>
                  <span className="row-sub">
                    {d.payer_country ?? "??"} · {formatTime(d.created_at)} ·{" "}
                    {d.source}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="detail">
          {selected ? (
            <>
              <h2>
                {selected.amount} {selected.currency} · {selected.decision}
              </h2>
              <p className="meta">
                {selected.id} · {formatTime(selected.created_at)} · payer{" "}
                {selected.payer_email ?? "unknown"} (
                {selected.payer_country ?? "unknown"}) · via {selected.source}
              </p>
              <div className="score-line">
                <span className={`badge big ${scoreClass(selected.risk_score)}`}>
                  {selected.risk_score.toFixed(2)} / 100
                </span>
                <span>
                  Recommendation: <strong>{selected.recommendation}</strong>
                </span>
              </div>

              <h3>Why</h3>
              <p>{selected.explanation}</p>
              <p className="meta">
                Explanation source: {selected.explanation_source}
                {selected.explanation_source === "template" &&
                  " (rule-based text, no LLM was used)"}
              </p>

              <h3>Top contributing features</h3>
              {selected.top_features.length > 0 ? (
                <ul className="features">
                  {selected.top_features.map((f) => (
                    <li key={f}>
                      <code>{f}</code>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="meta">No feature contributions recorded.</p>
              )}

              <h3>Signals</h3>
              <p>
                <strong>Real PayPal fields:</strong>{" "}
                {(selected.signals.real_paypal_signals ?? []).join(", ") ||
                  "none"}
              </p>
              <p>
                <strong>Defaulted (filled from training data):</strong>{" "}
                {(selected.signals.defaulted_signals ?? []).join(", ") ||
                  "none"}
              </p>

              {canAct && (
                <div className="actions">
                  <button
                    className="approve"
                    disabled={acting}
                    onClick={() => void act("approve")}
                  >
                    Approve
                  </button>
                  <button
                    className="block"
                    disabled={acting}
                    onClick={() => void act("block")}
                  >
                    Block
                  </button>
                </div>
              )}
              {selected.merchant_action && (
                <p className="meta">
                  Merchant action: {selected.merchant_action} at{" "}
                  {selected.action_at && formatTime(selected.action_at)}
                </p>
              )}
            </>
          ) : (
            <p className="empty">Select a transaction to see the detail.</p>
          )}
        </section>
      </main>
    </div>
  );
}
