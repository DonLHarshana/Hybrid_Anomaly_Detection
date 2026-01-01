#!/usr/bin/env python3
# ml/src/evaluate.py

import json, os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)

from features import FEATURE_COLS, build_feature_matrix


def safe_div(n, d):
    return float(n) / float(d) if d != 0 else 0.0


def find_best_threshold(y_true, scores):
    thresholds = np.percentile(scores, np.linspace(50, 99.9, 200))
    best_f1, best_thr, best_pred = 0.0, float(thresholds[0]), None
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr, best_pred = float(f1), float(thr), pred
    return best_pred, best_thr, best_f1


def main():
    model_path = Path("ml/models/isolation_forest_model_v1.pkl")
    scaler_path = Path("ml/models/scaler_v1.pkl")
    meta_path = Path("ml/models/isolation_forest_v1.meta.json")
    eval_path = Path("ml/data/eval.csv")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    expected_features = meta.get("expected_features") or FEATURE_COLS

    df = pd.read_csv(eval_path)

    # labels
    if "is_fraud" in df.columns:
        y_true = df["is_fraud"].astype(int).values
    elif "label" in df.columns:
        y_true = df["label"].astype(int).values
    elif "Class" in df.columns:
        y_true = df["Class"].astype(int).values
    else:
        raise ValueError("No label column found in eval.csv")

    # Build features consistently (creates missing engineered columns)
    X = build_feature_matrix(df)

    # Align to expected_features from meta
    for feat in expected_features:
        if feat not in X.columns:
            X[feat] = 0.0
    X_eval = X[expected_features].values

    if scaler is not None:
        X_eval = scaler.transform(X_eval)

    raw_scores = model.decision_function(X_eval)
    scores = -raw_scores
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    y_pred, best_thr, best_f1 = find_best_threshold(y_true, scores_norm)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    anomaly_rate = float(y_pred.mean())
    specificity = safe_div(tn, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)

    try:
        roc_auc = roc_auc_score(y_true, scores_norm)
        pr_auc = average_precision_score(y_true, scores_norm)
    except Exception:
        roc_auc, pr_auc = 0.0, 0.0

    mcc = matthews_corrcoef(y_true, y_pred)

    ml_metrics = {
        "anomaly_rate": round(anomaly_rate, 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }

    full_metrics = {
        **ml_metrics,
        "accuracy": round(float(accuracy), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 6),
        "specificity": round(float(specificity), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "mcc": round(float(mcc), 6),
        "best_threshold": round(float(best_thr), 6),
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "n_features": int(len(expected_features)),
    }

    os.makedirs("ml_out", exist_ok=True)
    with open("ml_out/ml_metrics.json", "w") as f:
        json.dump(ml_metrics, f, indent=2)
    with open("ml_out/ml_full_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    pd.DataFrame({
        "anomaly_score": scores_norm,
        "prediction": y_pred,
        "ground_truth": y_true
    }).to_csv("ml_out/ml_predictions.csv", index=False)

    print(json.dumps(ml_metrics, indent=2))


if __name__ == "__main__":
    main()
