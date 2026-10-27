"""Retrain the PayGuard AI risk model from the original fraud-gateway repo's
engineered dataset, replicating notebooks/03_Production_Pipeline.ipynb.

Why retrain instead of reusing the trained artifact?
The original repo's trained files (backend/model/fraud_pipeline.pkl and
frequency_maps.pkl) were never committed — its .gitignore excludes
``backend/model/*.pkl``. The pipeline recipe, hyperparameters, split seed and
source dataset are all public in that repo, so this script rebuilds the same
pipeline from the same data and saves text-only artifacts (no pickle) that
the FastAPI app can load:

  model/xgb_model.json      — XGBoost booster (native JSON format)
  model/encoding.json       — frequency maps, one-hot categories, feature order,
                              PayPal-mapping defaults and thresholds
  model/metrics.json        — holdout metrics of THIS retrained model
  model/config.json         — risk thresholds / decision labels (lifted as-is
                              from the original backend/config/config.json)

Two deviations from the notebook, both deliberate and documented:

1. The notebook recomputed ``frequency_maps`` from X_train AFTER the
   frequency-encoding step had already replaced the raw values, so the saved
   maps were keyed by numeric frequencies instead of raw merchant/city/job
   names (inference lookups of raw names would then miss and fall back to 0).
   Here the maps are computed from the RAW training values, before encoding —
   consistent with what the model was actually trained on.
2. The notebook's ``fillna(0, inplace=True)`` chained assignment did not take
   effect under pandas copy-on-write. Here the fill is a real assignment.

Usage:
    python scripts/train_model.py [--data PATH] [--out DIR]

If --data is omitted, the script looks for ./data/fraud_engineered.csv, then
~/.cache/payguard-ai/fraud_engineered.csv, then downloads the public dataset
from the original repo's raw GitHub URL into ./data/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA_URL = (
    "https://raw.githubusercontent.com/NabiAdnan/"
    "AI-Powered-Payment-Gateway-with-Real-Time-Fraud-Detection-and-Risk-Scoring/"
    "main/data/fraud_engineered.csv"
)
# The notebook pinned this commit; the dataset is identical at main.
DATA_URL_PINNED = (
    "https://raw.githubusercontent.com/NabiAdnan/"
    "AI-Powered-Payment-Gateway-with-Real-Time-Fraud-Detection-and-Risk-Scoring/"
    "66927be173337351a7d06d33d1c7770670e2611c/data/fraud_engineered.csv"
)

DROP_COLS = [
    "Customer_Satisfaction_Score",
    "Loyalty_Points_Earned",
    "Transaction_Time",
    "Favorite_Merchant",
    "Favorite_Category",
    "trans_date_trans_time",
    "dob",
    "Transaction_Day",
    "Transaction_Month",
    "Transaction_Number",
    "Customer_Max_Spend",
    "Customer_Total_Spend",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "cc_num",
]

HIGH_CARDINALITY = ["merchant", "city", "job"]


def resolve_data(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    candidates = [
        Path("data/fraud_engineered.csv"),
        Path.home() / ".cache" / "payguard-ai" / "fraud_engineered.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    target = Path("data/fraud_engineered.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading engineered dataset from {DATA_URL_PINNED} ...")
    urllib.request.urlretrieve(DATA_URL_PINNED, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PayGuard AI risk model.")
    parser.add_argument("--data", default=None, help="Path to fraud_engineered.csv")
    parser.add_argument("--out", default="model", help="Output directory for artifacts")
    args = parser.parse_args()

    data_path = resolve_data(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    feature_columns = X.columns.tolist()

    # Frequency-encode high-cardinality columns from RAW training values.
    frequency_maps: dict[str, dict[str, float]] = {}
    for col in HIGH_CARDINALITY:
        freq = X_train[col].value_counts(normalize=True)
        frequency_maps[col] = {str(k): float(v) for k, v in freq.items()}
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[col] = X_train[col].map(freq).fillna(0)
        X_test[col] = X_test[col].map(freq).fillna(0)

    categorical_features = X_train.select_dtypes(include="object").columns.tolist()
    numeric_features = [c for c in feature_columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "fraud_rate": round(float(y.mean()), 4),
    }
    print("Holdout metrics:", json.dumps(metrics, indent=2))

    # ---- Text-only artifacts (no pickle) ----
    booster = pipeline.named_steps["classifier"].get_booster()
    booster.save_model(str(out_dir / "xgb_model.json"))

    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    categories = {
        feat: [str(c) for c in cats]
        for feat, cats in zip(categorical_features, encoder.categories_)
    }

    # Defaults/thresholds used by the PayPal feature builder, derived from the
    # training data so they are data-driven, not invented.
    amt = X_train["amt"]
    defaults = {
        "city_pop": float(X_train["city_pop"].median()),
        "Customer_Age": float(X_train["Customer_Age"].median()),
        "Customer_Avg_Spend": None,  # filled per-request from the amount
        "High_Amount_Flag_threshold": float(amt.quantile(0.95)),
    }

    encoding = {
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "categories": categories,
        "frequency_maps": frequency_maps,
        "defaults": defaults,
        "target": "is_fraud",
    }
    (out_dir / "encoding.json").write_text(json.dumps(encoding, indent=2))

    provenance = {
        **metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "fraud_engineered.csv from NabiAdnan/AI-Powered-Payment-Gateway-with-Real-Time-Fraud-Detection-and-Risk-Scoring",
        "recipe": "notebooks/03_Production_Pipeline.ipynb (same split seed, SMOTE seed, XGBoost hyperparameters)",
        "note": (
            "Retrained because the original .pkl artifacts were never committed "
            "(gitignored). Metrics above are this retrained model's holdout "
            "results, not the original model's reported figures."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(provenance, indent=2))
    print(f"Artifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
