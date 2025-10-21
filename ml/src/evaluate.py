# ml/src/evaluate.py
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from utils import ensure_dir, write_json, log, read_json

# Optional import to detect sklearn Pipeline
try:
    from sklearn.pipeline import Pipeline as SkPipeline
except Exception:  # fallback if sklearn import path differs
    SkPipeline = tuple()

MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")
META_PATH  = Path("ml/models/isolation_forest_v1.meta.json")
EVAL_PATH  = Path("ml/data/eval.csv")
OUT_DIR    = ensure_dir("ml_out")
OUT_METRICS = OUT_DIR / "ml_metrics.json"
OUT_PRED    = OUT_DIR / "ml_predictions.csv"

def resolve_feature_order(expected_from_meta, model):
    if hasattr(model, "feature_names_in_") and len(getattr(model, "feature_names_in_")) > 0:
        names = list(model.feature_names_in_)
        log(f"Using feature_names_in_ from model (n={len(names)})")
        return names
    n_model = getattr(model, "n_features_in_", None)
    if n_model is None:
        log("Model lacks n_features_in_; falling back to meta expected_features")
        return list(expected_from_meta)
    exp = list(expected_from_meta)
    if len(exp) == n_model:
        return exp
    if len(exp) > n_model:
        trimmed = exp[:n_model]; removed = exp[n_model:]
        log(f"WARNING: meta has {len(exp)}, model expects {n_model}. Trimming extras: {removed}")
        return trimmed
    missing_count = n_model - len(exp)
    pads = [f"_pad_{i}" for i in range(missing_count)]
    log(f"WARNING: meta has {len(exp)}, model expects {n_model}. Padding with zeros: {pads}")
    return exp + pads

def align(df: pd.DataFrame, ordered_names, fill_value=0.0) -> pd.DataFrame:
    for col in ordered_names:
        if col not in df.columns:
            df[col] = fill_value
    return df[ordered_names]

def _confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn

def _safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return None

def _pick_label_column(df: pd.DataFrame):
    for name in ["label", "Fraud_Flag", "is_fraud", "fraud"]:
        if name in df.columns:
            return name
    return None

def main():
    log("Loading model")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    log("Loading meta")
    meta = read_json(META_PATH) or {}
    expected_from_meta = meta.get("expected_features", [])

    log(f"Loading eval: {EVAL_PATH}")
    df = pd.read_csv(EVAL_PATH)

    # --- Pick a label if present (used only for metrics) ---
    label_col = _pick_label_column(df)
    y = df[label_col].astype(int).values if label_col else None
    if label_col:
        log(f"Using label column: {label_col}")
    else:
        log("No label column found; metrics will omit supervised scores.")

    # --- Predict path depends on model type ---
    is_pipeline = isinstance(model, SkPipeline)

    if is_pipeline:
        Xdf = df.copy()  # raw columns; pipeline does feats+encoding+scaling
        log("Predicting with Pipeline (raw columns)")
        pred_raw = model.predict(Xdf)            # +1 normal, -1 anomaly
        pred     = (pred_raw == -1).astype(int)  # 1 anomaly, 0 normal
        scores   = -model.score_samples(Xdf)     # higher => more anomalous

        # Feature counts (after preprocessing)
        try:
            X_trans = model[:-1].transform(Xdf)
            n_features_eval = int(X_trans.shape[1])
        except Exception:
            n_features_eval = int(len(expected_from_meta)) if expected_from_meta else None
        n_features_model_expected = int(len(expected_from_meta)) if expected_from_meta else n_features_eval
        n_samples = int(df.shape[0])

    else:
        # Legacy numeric path (for old plain IF models)
        ordered_features = resolve_feature_order(expected_from_meta, model)
        Xdf = align(df.copy(), ordered_features).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        log("Predicting with legacy numeric path")
        pred_raw = model.predict(Xdf)            # +1 normal, -1 anomaly (or 1/-1 depending on impl)
        pred     = (pred_raw == -1).astype(int)
        scores   = -model.score_samples(Xdf)
        n_features_eval = int(Xdf.shape[1])
        n_features_model_expected = int(getattr(model, "n_features_in_", Xdf.shape[1]))
        n_samples = int(Xdf.shape[0])

    # --- Aggregate metrics ---
    metrics = {
        "anomaly_rate": float(np.mean(pred)),
        "mean_anomaly_score": float(np.mean(scores)),
        "n_features_eval": n_features_eval if n_features_eval is not None else 0,
        "n_features_model_expected": n_features_model_expected if n_features_model_expected is not None else 0,
        "n_samples": n_samples,
    }

    if y is not None:
        metrics["n_positive"] = int(np.sum(y == 1))
        metrics["n_negative"] = int(np.sum(y == 0))

        tp, fp, tn, fn = _confusion(y, pred)
        metrics["TP"] = tp; metrics["FP"] = fp; metrics["TN"] = tn; metrics["FN"] = fn

        metrics["precision"] = _safe_metric(precision_score, y, pred, zero_division=0)
        metrics["recall"]    = _safe_metric(recall_score,    y, pred, zero_division=0)
        metrics["f1"]        = _safe_metric(f1_score,        y, pred, zero_division=0)
        metrics["accuracy"]  = _safe_metric(accuracy_score,  y, pred)
        metrics["balanced_accuracy"] = _safe_metric(balanced_accuracy_score, y, pred)
        metrics["specificity"] = (float(tn) / float(tn + fp)) if (tn + fp) > 0 else None
        metrics["mcc"] = _safe_metric(matthews_corrcoef, y, pred)

        if np.any(y == 0) and np.any(y == 1):
            metrics["roc_auc"] = _safe_metric(roc_auc_score, y, scores)
            metrics["pr_auc"]  = _safe_metric(average_precision_score, y, scores)
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"]  = None

        # Threshold sweep to report best F1
        uniq = np.unique(scores)
        cand = uniq[np.linspace(0, uniq.size - 1, min(400, uniq.size)).astype(int)]
        best = {"f1": -1.0, "threshold": None, "metrics": None}
        for t in cand:
            pred_t = (scores >= t).astype(int)
            tp_t, fp_t, tn_t, fn_t = _confusion(y, pred_t)
            f1_t = _safe_metric(f1_score, y, pred_t, zero_division=0)
            if f1_t is not None and f1_t > best["f1"]:
                best = {
                    "f1": f1_t, "threshold": float(t),
                    "metrics": {
                        "TP": tp_t, "FP": fp_t, "TN": tn_t, "FN": fn_t,
                        "precision": _safe_metric(precision_score, y, pred_t, zero_division=0),
                        "recall": _safe_metric(recall_score, y, pred_t, zero_division=0),
                        "f1": f1_t,
                    }
                }
        metrics["best_f1_threshold"] = best

    write_json(OUT_METRICS, metrics)

    df_out = df.copy()
    df_out["prediction"] = pred
    df_out["anomaly_score"] = scores
    df_out.to_csv(OUT_PRED, index=False)

    log(json.dumps(metrics))

if __name__ == "__main__":
    main()
