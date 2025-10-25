#!/usr/bin/env python3
# ml/src/evaluate.py
"""
Evaluate Isolation Forest model on test data and output metrics.
Generates both simplified (for CI/CD) and full (for thesis) metric files.
"""
import json
import os
import joblib  # ✅ Changed from pickle to joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)


def safe_div(n, d):
    """Safe division to avoid division by zero"""
    return float(n) / float(d) if d != 0 else 0.0


def decide_predictions(scores, contamination_rate=0.031, mode="quantile"):
    """
    Convert anomaly scores to binary predictions
    
    Args:
        scores: array of anomaly scores
        contamination_rate: expected proportion of anomalies
        mode: "quantile" or "threshold"
    
    Returns:
        Binary predictions (1 = anomaly, 0 = normal)
    """
    if mode == "quantile":
        threshold = np.quantile(scores, 1 - contamination_rate)
        return (scores >= threshold).astype(int)
    elif mode == "threshold":
        return (scores > 0.5).astype(int)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_balanced_accuracy(tn, fp, fn, tp):
    """Compute balanced accuracy from confusion matrix"""
    sensitivity = safe_div(tp, tp + fn)  # recall
    specificity = safe_div(tn, tn + fp)
    return (sensitivity + specificity) / 2.0


def find_best_f1_threshold(y_true, scores):
    """Find threshold that maximizes F1 score"""
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            best_metrics = {
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1)
            }
    
    return {
        "f1": best_f1,
        "threshold": best_threshold,
        "metrics": best_metrics
    }


def main():
    # Paths
    model_path = Path("ml/models/isolation_forest_model_v1.pkl")
    meta_path = Path("ml/models/isolation_forest_v1.meta.json")
    eval_data_path = Path("ml/data/eval.csv")
    
    # ✅ Load model using joblib instead of pickle
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    contamination = meta.get("best_params", {}).get("contamination", 0.031)
    expected_features = meta.get("expected_features", [])
    
    # Load evaluation data
    print(f"Loading evaluation data from {eval_data_path}")
    df_eval = pd.read_csv(eval_data_path)
    
    # Separate features and labels
    if "is_fraud" in df_eval.columns:
        y_true = df_eval["is_fraud"].values
    elif "label" in df_eval.columns:
        y_true = df_eval["label"].values
    else:
        raise ValueError("eval.csv must contain either 'is_fraud' or 'label' column")

    X_eval = df_eval[expected_features].values
    
    # Generate predictions
    print("Generating predictions...")
    anomaly_scores = model.decision_function(X_eval)
    anomaly_scores_normalized = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    y_pred = decide_predictions(anomaly_scores_normalized, contamination, mode="quantile")
    
    # Calculate core metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    anomaly_rate = y_pred.sum() / len(y_pred)
    
    # Calculate extended metrics (for thesis)
    balanced_accuracy = compute_balanced_accuracy(tn, fp, fn, tp)
    specificity = safe_div(tn, tn + fp)
    
    try:
        roc_auc = roc_auc_score(y_true, anomaly_scores_normalized)
    except:
        roc_auc = 0.0
    
    try:
        pr_auc = average_precision_score(y_true, anomaly_scores_normalized)
    except:
        pr_auc = 0.0
    
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Find best F1 threshold
    best_f1_result = find_best_f1_threshold(y_true, anomaly_scores_normalized)
    
    # ============================================
    # SIMPLIFIED METRICS (for Decision Gate)
    # ============================================
    ml_metrics = {
        "anomaly_rate": round(float(anomaly_rate), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }
    
    # ============================================
    # FULL METRICS (for Thesis Evaluation)
    # ============================================
    full_metrics = {
        **ml_metrics,
        "accuracy": round(float(accuracy), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 6),
        "specificity": round(float(specificity), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "mcc": round(float(mcc), 6),
        "mean_anomaly_score": round(float(anomaly_scores_normalized.mean()), 6),
        "n_samples": int(len(y_true)),
        "n_features_eval": int(len(expected_features)),
        "contamination": float(contamination),
        "decision_mode": "quantile",
        "best_f1_threshold": best_f1_result
    }
    
    # Create output directory
    os.makedirs("ml_out", exist_ok=True)
    
    # Save simplified metrics (for CI/CD Decision Gate)
    with open("ml_out/ml_metrics.json", "w") as f:
        json.dump(ml_metrics, f, indent=2)
    
    # Save full metrics (for Thesis Chapter 5)
    with open("ml_out/ml_full_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    
    # Save predictions CSV (for detailed analysis)
    predictions_df = pd.DataFrame({
        "anomaly_score": anomaly_scores_normalized,
        "prediction": y_pred,
        "ground_truth": y_true
    })
    predictions_df.to_csv("ml_out/ml_predictions.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ ML EVALUATION COMPLETE")
    print("="*60)
    print(f"📊 Simplified metrics → ml_out/ml_metrics.json")
    print(f"📈 Full metrics → ml_out/ml_full_metrics.json")
    print(f"📋 Predictions → ml_out/ml_predictions.csv")
    print("="*60)
    print("\n### Simplified ML Metrics (for Decision Gate)")
    print(json.dumps(ml_metrics, indent=2))
    print("\n")


if __name__ == "__main__":
    main()
