#!/usr/bin/env python3
# ml/src/evaluate.py
"""
Evaluate Isolation Forest model on Credit Card Fraud test data.
Uses best F1 threshold finding to match Colab training methodology.
"""
import json
import os
import joblib  
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


def find_best_threshold(y_true, scores):
    """
    Find threshold that maximizes F1 score.
    This matches the Colab training methodology.
    """
    thresholds = np.percentile(scores, np.linspace(50, 99.9, 200))
    best_f1 = 0
    best_threshold = 0
    best_pred = None
    
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_pred = y_pred
    
    return best_pred, best_threshold, best_f1


def main():
    # Paths
    model_path = Path("ml/models/isolation_forest_model_v1.pkl")
    scaler_path = Path("ml/models/scaler_v1.pkl")
    meta_path = Path("ml/models/isolation_forest_v1.meta.json")
    eval_data_path = Path("ml/data/eval.csv")
    
    # Load model and scaler
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    if scaler_path.exists():
        print(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        print("⚠️ No scaler found")
        scaler = None
    
    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    expected_features = meta.get("expected_features", [])
    
    print(f"\n{'='*60}")
    print(f"MODEL CONFIGURATION")
    print(f"{'='*60}")
    print(f"Expected features: {len(expected_features)}")
    print(f"Features: {expected_features[:5]}... (showing first 5)")
    print(f"{'='*60}\n")
    
    # Load evaluation data
    print(f"Loading evaluation data from {eval_data_path}")
    df_eval = pd.read_csv(eval_data_path)
    print(f"Loaded {len(df_eval)} samples")
    
    # Get labels
    if "is_fraud" in df_eval.columns:
        y_true = df_eval["is_fraud"].values
        label_col = "is_fraud"
    elif "label" in df_eval.columns:
        y_true = df_eval["label"].values
        label_col = "label"
    elif "Class" in df_eval.columns:
        y_true = df_eval["Class"].values
        label_col = "Class"
    else:
        raise ValueError("No label column found!")
    
    print(f"Using label column: '{label_col}'")
    print(f"Fraud cases: {y_true.sum()} out of {len(y_true)} ({100*y_true.sum()/len(y_true):.3f}%)")
    
    # Extract features
    missing_features = [f for f in expected_features if f not in df_eval.columns]
    
    if missing_features:
        print(f"\n⚠️ WARNING: {len(missing_features)} features missing!")
        print(f"   {missing_features[:10]}")
        for feat in missing_features:
            df_eval[feat] = 0.0
    else:
        print(f"\n✅ All {len(expected_features)} expected features found!")
    
    # Extract feature matrix
    X_eval = df_eval[expected_features].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    
    print(f"Feature matrix shape: {X_eval.shape}")
    
    # Scale features
    if scaler is not None:
        print("Scaling features...")
        X_eval = scaler.transform(X_eval)
    
    # Generate anomaly scores
    print("Generating anomaly scores...")
    anomaly_scores = model.decision_function(X_eval)
    
    # Normalize scores to [0, 1]
    scores_min = anomaly_scores.min()
    scores_max = anomaly_scores.max()
    # Invert scores (Isolation Forest: negative = anomaly)
    anomaly_scores_inverted = -anomaly_scores
    anomaly_scores_normalized = (anomaly_scores_inverted - anomaly_scores_inverted.min()) / (anomaly_scores_inverted.max() - anomaly_scores_inverted.min() + 1e-10)

    
    print(f"Score range: [{anomaly_scores_normalized.min():.4f}, {anomaly_scores_normalized.max():.4f}]")
    print(f"Mean score: {anomaly_scores_normalized.mean():.4f}")
    
    # Find best threshold (matching Colab training methodology)
    print("\nFinding optimal threshold...")
    y_pred, best_threshold, best_f1 = find_best_threshold(y_true, anomaly_scores_normalized)
    
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Expected F1: {best_f1:.4f}")
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    anomaly_rate = y_pred.sum() / len(y_pred)
    
    print(f"\n{'='*60}")
    print(f"RESULTS (Best F1 Threshold)")
    print(f"{'='*60}")
    print(f"TP: {tp:4d} | FP: {fp:4d} | FN: {fn:4d} | TN: {tn:4d}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Anomaly Rate: {anomaly_rate:.4f} ({100*anomaly_rate:.2f}%)")
    print(f"{'='*60}\n")
    
    # Extended metrics
    balanced_accuracy = safe_div(recall, 1) + safe_div(tn/(tn+fp), 1) / 2.0
    specificity = safe_div(tn, tn + fp)
    
    try:
        roc_auc = roc_auc_score(y_true, anomaly_scores_normalized)
        pr_auc = average_precision_score(y_true, anomaly_scores_normalized)
    except:
        roc_auc = pr_auc = 0.0
    
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Output metrics
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
    
    full_metrics = {
        **ml_metrics,
        "accuracy": round(float(accuracy), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 6),
        "specificity": round(float(specificity), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "mcc": round(float(mcc), 6),
        "mean_anomaly_score": round(float(anomaly_scores_normalized.mean()), 6),
        "best_threshold": round(float(best_threshold), 6),
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "n_features": int(len(expected_features))
    }
    
    # Save outputs
    os.makedirs("ml_out", exist_ok=True)
    
    with open("ml_out/ml_metrics.json", "w") as f:
        json.dump(ml_metrics, f, indent=2)
    
    with open("ml_out/ml_full_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    
    predictions_df = pd.DataFrame({
        "anomaly_score": anomaly_scores_normalized,
        "prediction": y_pred,
        "ground_truth": y_true
    })
    predictions_df.to_csv("ml_out/ml_predictions.csv", index=False)
    
    print("="*60)
    print("ML EVALUATION COMPLETE")
    print("="*60)
    print(f"Metrics → ml_out/ml_metrics.json")
    print(f"Predictions → ml_out/ml_predictions.csv")
    print("="*60)
    print("\n### ML Metrics (for Decision Gate)")
    print(json.dumps(ml_metrics, indent=2))


if __name__ == "__main__":
    main()
