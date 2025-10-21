# ml/src/train.py
from pathlib import Path
import os, json, joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from utils import ensure_dir, log, write_json
from features import FEATURE_COLS

MODEL_DIR  = ensure_dir("ml/models")
MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")
META_PATH  = Path("ml/models/isolation_forest_v1.meta.json")
TRAIN_CSV  = Path("ml/data/train.csv")

def main(contamination=0.10, random_state=42, n_estimators=200):
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"{TRAIN_CSV} not found")

    log(f"Loading {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)

    # lock feature order & ensure presence
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    Xdf = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    log("Training Pipeline[StandardScaler -> IsolationForest]")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iso", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    pipe.fit(Xdf)  # keeps feature names inside pipeline

    joblib.dump(pipe, MODEL_PATH)

    # Save exactly what was used (robust to drift in FEATURE_COLS)
    used_features = list(getattr(pipe, "feature_names_in_", Xdf.columns.tolist()))
    write_json(META_PATH, {
        "expected_features": used_features,
        "model": "IsolationForest",
        "preprocessing": ["StandardScaler"],
        "params": {"n_estimators": n_estimators, "contamination": contamination, "random_state": random_state},
        "version": "v1"
    })

    log(f"Saved model -> {MODEL_PATH}")
    log(f"Saved meta  -> {META_PATH}")

if __name__ == "__main__":
    main(
        contamination=float(os.getenv("IF_CONTAMINATION", "0.10")),
        random_state=int(os.getenv("IF_SEED", "42")),
        n_estimators=int(os.getenv("IF_TREES", "200")),
    )
