/** API client for the PayGuard AI backend.
 * Backend URL comes from VITE_API_URL (build time) and defaults to the
 * local backend on :8000. See the repo README for run steps. */

export interface Decision {
  id: string;
  created_at: string;
  amount: string;
  currency: string;
  payer_country: string | null;
  payer_email: string | null;
  risk_score: number;
  decision: string;
  explanation: string;
  explanation_source: string;
  recommendation: string;
  signals: { real_paypal_signals?: string[]; defaulted_signals?: string[] };
  top_features: string[];
  source: string;
  merchant_action: string | null;
  action_at: string | null;
}

const API_BASE =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(
      (detail as { detail?: string }).detail ?? `Request failed: ${response.status}`,
    );
  }
  return response.json() as Promise<T>;
}

export function fetchDecisions(limit = 50): Promise<Decision[]> {
  return request<Decision[]>(`/decisions?limit=${limit}`);
}

export function recordAction(
  id: string,
  action: "approve" | "block",
): Promise<Decision> {
  return request<Decision>(`/decisions/${id}/action`, {
    method: "POST",
    body: JSON.stringify({ action }),
  });
}
