# ml/src/train.py
from pathlib import Path
import numpy as np
import pandas as pd
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

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    Xdf = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    log("Training IsolationForest")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(Xdf)  # ensures model.feature_names_in_ is set

    import joblib
    joblib.dump(model, MODEL_PATH)
    write_json(META_PATH, {"expected_features": FEATURE_COLS})

    log(f"Saved model -> {MODEL_PATH}")
    log(f"Saved meta  -> {META_PATH}")

if __name__ == "__main__":
    import os
    main(
        contamination=float(os.getenv("IF_CONTAMINATION", "0.10")),
        random_state=int(os.getenv("IF_SEED", "42")),
        n_estimators=int(os.getenv("IF_TREES", "200")),
    )
