# ml/src/train.py
from pathlib import Path
import os, joblib, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import IsolationForest
from utils import ensure_dir, log, write_json
from features import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, DATE_TIME, ID_OR_DROP, FEATURE_COLS
)

MODEL_DIR  = ensure_dir("ml/models")
MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")
META_PATH  = Path("ml/models/isolation_forest_v1.meta.json")
TRAIN_CSV  = Path("ml/data/train.csv")

LABEL_NAMES = {"label","is_fraud","fraud","Fraud_Flag"}

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer simple time features from Transaction_Date/Time."""
    out = df.copy()
    dt = pd.to_datetime(out.get("Transaction_Date"), errors="coerce")
    tt = pd.to_datetime(out.get("Transaction_Time"), format="%H:%M:%S", errors="coerce")
    out["txn_hour"]    = tt.dt.hour.fillna(0).astype(int)
    out["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
    out["is_weekend"]  = out["day_of_week"].isin([5, 6]).astype(int)
    return out

def main(contamination=0.03, random_state=42, n_estimators=200):
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"{TRAIN_CSV} not found")

    log(f"Loading {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)

    # Safety: ensure label is not accidentally in feature lists 
    assert not (LABEL_NAMES & set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)), \
        f"Label leaked into features: {LABEL_NAMES & set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)}"

    # Ensure referenced columns exist (fill missing so the pipeline won’t crash)
    for c in set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + DATE_TIME):
        if c not in df.columns:
            df[c] = np.nan

    # ColumnTransformer for preprocessing:
    # - Standardize numeric (plus engineered time features)
    # - One-hot encode categoricals (dense array)
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + ["txn_hour", "day_of_week", "is_weekend"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Full pipeline: add time feats -> preprocess -> Isolation Forest algorithm
    pipe = Pipeline([
        ("feats", FunctionTransformer(add_time_features, validate=False)),
        ("prep",  pre),
        ("iso",   IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,   # ~3% in your dataset; tune 0.03–0.05
            random_state=random_state,
            n_jobs=-1
        )),
    ])

    log("Fitting pipeline")
    pipe.fit(df)  # fit on raw columns; pipeline handles all transforms

    joblib.dump(pipe, MODEL_PATH)

    # Persist the post-encoding feature names (what the model actually sees)
    used_features = list(pipe.named_steps["prep"].get_feature_names_out())

    write_json(META_PATH, {
        "raw_feature_spec": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "engineered_time": ["txn_hour", "day_of_week", "is_weekend"],
            "dropped": ID_OR_DROP + DATE_TIME
        },
        "expected_features": used_features,  # after one-hot + scaling
        "model": "IsolationForest",
        "preprocessing": ["FunctionTransformer(time_features)", "StandardScaler(num)", "OneHot(cat)"],
        "params": {"n_estimators": n_estimators, "contamination": contamination, "random_state": random_state},
        "version": "v1"
    })

    log(f"Saved model -> {MODEL_PATH}")
    log(f"Saved meta  -> {META_PATH}")

if __name__ == "__main__":
    main(
        contamination=float(os.getenv("IF_CONTAMINATION", "0.03")),  # ≈ your ~3.1% fraud rate
        random_state=int(os.getenv("IF_SEED", "42")),
        n_estimators=int(os.getenv("IF_TREES", "200")),
    )
