"""Model loading and scoring.

The model is stored as text-only artifacts produced by scripts/train_model.py
(no pickle): an XGBoost booster in native JSON plus an encoding manifest with
frequency maps, one-hot categories and feature order. This module rebuilds
the exact feature vector the training pipeline produced:

  [one-hot blocks for each categorical feature, in training order]
  + [passthrough numeric features, in training column order]

with merchant/city/job frequency-encoded first, exactly as in training.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import xgboost as xgb

from .config import MODEL_DIR

_lock = threading.Lock()
_booster: xgb.Booster | None = None
_encoding: dict | None = None
_load_error: str | None = None


def _load() -> None:
    global _booster, _encoding, _load_error
    if _booster is not None or _load_error is not None:
        return
    with _lock:
        if _booster is not None or _load_error is not None:
            return
        try:
            model_path = MODEL_DIR / "xgb_model.json"
            encoding_path = MODEL_DIR / "encoding.json"
            if not model_path.exists() or not encoding_path.exists():
                raise FileNotFoundError(
                    "Model artifacts not found. Run: python scripts/train_model.py"
                )
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            _encoding = json.loads(encoding_path.read_text())
            _booster = booster
        except Exception as exc:  # noqa: BLE001 — surfaced via health endpoint
            _load_error = str(exc)


def is_loaded() -> bool:
    _load()
    return _booster is not None


def load_error() -> str | None:
    _load()
    return _load_error


def encoding() -> dict:
    _load()
    if _encoding is None:
        raise RuntimeError(_load_error or "Model artifacts not loaded")
    return _encoding


def metrics() -> dict:
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


def build_vector(features: dict) -> np.ndarray:
    """Build the model input vector from a dict of the 35 raw feature fields."""
    enc = encoding()
    row: list[float] = []

    values = dict(features)
    for col, freq_map in enc["frequency_maps"].items():
        raw = values.get(col)
        values[col] = float(freq_map.get(str(raw), 0.0)) if raw is not None else 0.0

    for feat in enc["categorical_features"]:
        cats = enc["categories"][feat]
        val = str(values.get(feat, ""))
        row.extend(1.0 if val == cat else 0.0 for cat in cats)

    for feat in enc["numeric_features"]:
        val = values.get(feat, 0)
        row.append(float(val) if val is not None else 0.0)

    return np.asarray([row], dtype=np.float64)


def score_transaction(features: dict) -> dict:
    """Score one transaction. Returns prediction + fraud probability (0-1)."""
    _load()
    if _booster is None:
        raise RuntimeError(_load_error or "Model artifacts not loaded")
    vector = build_vector(features)
    dmatrix = xgb.DMatrix(vector)
    proba = float(_booster.predict(dmatrix)[0])
    return {"prediction": int(proba >= 0.5), "fraud_probability": proba}
