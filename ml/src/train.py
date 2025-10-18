# ml/src/train.py
from pathlib import Path
import json, joblib, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import __version__ as sklearn_version
from utils import ensure_dir, log
from features import FEATURE_COLS   # <- your full list of 15 features

TRAIN_CSV = Path("ml/data/train.csv")
MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")
META_PATH  = Path("ml/models/isolation_forest_v1.meta.json")

def main():
    log("Loading training data")
    df = pd.read_csv(TRAIN_CSV)
    X = df[FEATURE_COLS].astype(float)

    log("Building pipeline")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iso", IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1))
    ])

    log("Fitting")
    pipe.fit(X)

    ensure_dir(MODEL_PATH.parent)
    joblib.dump(pipe, MODEL_PATH)

    meta = {
        "expected_features": FEATURE_COLS,
        "sklearn_version": sklearn_version,
        "model_path": str(MODEL_PATH),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    log(f"Saved model -> {MODEL_PATH}")
    log(f"Saved meta  -> {META_PATH}")

if __name__ == "__main__":
    main()
