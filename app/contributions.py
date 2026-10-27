"""Per-transaction feature contributions — what actually moved this score.

Method (stated honestly, here and in the README): XGBoost's native
per-prediction contributions (``Booster.predict(..., pred_contribs=True)``),
the tree-SHAP approximation built into XGBoost. No ``shap`` package is
installed, so no SHAP library output is claimed. Contributions are computed
on the expanded model vector (one-hot blocks + numeric columns) and then
aggregated back to the 35 raw feature names: a categorical feature gets the
sum of its one-hot block's contributions; a numeric feature gets its single
column's contribution. Contributions are in log-odds units and sum (plus the
bias term) to the model's logit for this transaction.

Fallback: if per-prediction contributions are unavailable for any reason,
global gain importances (``Booster.get_score``) are used instead and the
method is reported as ``xgb_gain_importance`` — a global ranking, not a
per-transaction one. The caller surfaces the method so a reader can tell
which kind of evidence a ``top_features`` list is.
"""

from __future__ import annotations

import xgboost as xgb

from .model import build_vector, encoding

METHOD_CONTRIBS = "xgb_pred_contribs"
METHOD_GAIN = "xgb_gain_importance"


def _column_map(enc: dict) -> list[tuple[str, int]]:
    """Map each expanded vector column back to its raw feature name.

    Mirrors ``model.build_vector`` exactly: one-hot blocks for each
    categorical feature (in training order), then the numeric features.
    """
    columns: list[tuple[str, int]] = []
    idx = 0
    for feat in enc["categorical_features"]:
        for _ in enc["categories"][feat]:
            columns.append((feat, idx))
            idx += 1
    for feat in enc["numeric_features"]:
        columns.append((feat, idx))
        idx += 1
    return columns


def _aggregate(
    per_column: list[float], columns: list[tuple[str, int]], features: dict
) -> list[dict]:
    """Aggregate per-column contributions to per-raw-feature contributions."""
    totals: dict[str, float] = {}
    for feat, idx in columns:
        totals[feat] = totals.get(feat, 0.0) + per_column[idx]
    ranked = sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [
        {
            "feature": feat,
            "value": str(features.get(feat)),
            "contribution": round(contrib, 4),
        }
        for feat, contrib in ranked
    ]


def display_value(value: str) -> str:
    """Short display form of a raw feature value (floats rounded to 2dp)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return f"{number:.2f}"


def top_contributions(features: dict, n: int = 3) -> tuple[list[dict], str]:
    """Top ``n`` raw features by absolute contribution for this transaction.

    Returns (ranked contributions, method). The caller decides how to
    present them; the method string is part of the result so it can be
    reported alongside the numbers.
    """
    enc = encoding()
    columns = _column_map(enc)
    vector = build_vector(features)
    booster = xgb.Booster()
    from .config import MODEL_DIR

    booster.load_model(str(MODEL_DIR / "xgb_model.json"))
    try:
        contribs = booster.predict(xgb.DMatrix(vector), pred_contribs=True)[0]
        # Last entry is the bias term; drop it before aggregating.
        return _aggregate([float(c) for c in contribs[:-1]], columns, features)[:n], METHOD_CONTRIBS
    except Exception:  # noqa: BLE001 — documented fallback to global importances
        gains = booster.get_score(importance_type="gain")
        per_column = [float(gains.get(f"f{i}", 0.0)) for i in range(len(columns))]
        return _aggregate(per_column, columns, features)[:n], METHOD_GAIN
