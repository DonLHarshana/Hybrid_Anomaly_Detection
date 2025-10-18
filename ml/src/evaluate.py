# ml/src/evaluate.py
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import ensure_dir, write_json, log, read_json

MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")
META_PATH  = Path("ml/models/isolation_forest_v1.meta.json")
EVAL_PATH  = Path("ml/data/eval.csv")
OUT_DIR    = ensure_dir("ml_out")
OUT_METRICS = OUT_DIR / "ml_metrics.json"
OUT_PRED    = OUT_DIR / "ml_predictions.csv"

def align(df: pd.DataFrame, expected_order, fill_value=0.0) -> pd.DataFrame:
    for col in expected_order:
        if col not in df.columns:
            df[col] = fill_value
    return df[expected_order]

def main():
    log("Loading model")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    log("Loading meta")
    meta = read_json(META_PATH)
    if not meta or "expected_features" not in meta:
        raise FileNotFoundError(f"Missing/invalid meta: {META_PATH}")
    expected = meta["expected_features"]

    log(f"Loading eval: {EVAL_PATH}")
    df = pd.read_csv(EVAL_PATH)
    y = df["label"].astype(int).values if "label" in df.columns else None

    X = align(df.copy(), expected).to_numpy(dtype=float)

    log("Predicting")
    pred_raw = model.predict(X)           # -1 anomaly, 1 normal
    pred = (pred_raw == -1).astype(int)
    scores = -model.score_samples(X)

    metrics = {}
    if y is not None:
        metrics["precision"] = float(precision_score(y, pred, zero_division=0))
        metrics["recall"]    = float(recall_score(y, pred, zero_division=0))
        metrics["f1"]        = float(f1_score(y, pred, zero_division=0))
    metrics["anomaly_rate"] = float(np.mean(pred))
    metrics["mean_anomaly_score"] = float(np.mean(scores))
    metrics["n_features_eval"] = int(X.shape[1])
    metrics["n_features_model_expected"] = int(len(expected))

    write_json(OUT_METRICS, metrics)
    df_out = df.copy()
    df_out["prediction"] = pred
    df_out["anomaly_score"] = scores
    df_out.to_csv(OUT_PRED, index=False)
    log(json.dumps(metrics))

if __name__ == "__main__":
    main()
