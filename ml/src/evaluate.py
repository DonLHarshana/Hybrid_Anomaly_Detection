#!/usr/bin/env python3
# ml/src/evaluate.py
"""
Evaluate Isolation Forest model on Credit Card Fraud test data.
Automatically creates engineered features if they don't exist.
Generates both simplified (for CI/CD) and full (for thesis) metric files.
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


def create_engineered_features(df):
    """
    Create engineered features for Credit Card Fraud dataset.
    Only creates features that don't already exist.
    
    Args:
        df: DataFrame with raw features
    
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Amount features
    if 'Amount' in df.columns:
        if 'Amount_log' not in df.columns:
            df['Amount_log'] = np.log1p(df['Amount'])
        if 'Amount_sqrt' not in df.columns:
            df['Amount_sqrt'] = np.sqrt(df['Amount'])
        if 'Amount_cube' not in df.columns:
            df['Amount_cube'] = np.cbrt(df['Amount'])
    
    # Time features
    if 'Time' in df.columns and 'Time_sin' not in df.columns:
        df['Time_sin'] = np.sin(2 * np.pi * df['Time'] / 86400)
        df['Time_cos'] = np.cos(2 * np.pi * df['Time'] / 86400)
    
    # V statistics (mean, std, max, min of V1-V28)
    v_cols = [f'V{i}' for i in range(1, 29) if f'V{i}' in df.columns]
    
    if v_cols and 'V_mean' not in df.columns:
        df['V_mean'] = df[v_cols].mean(axis=1)
        df['V_std'] = df[v_cols].std(axis=1)
        df['V_max'] = df[v_cols].max(axis=1)
        df['V_min'] = df[v_cols].min(axis=1)
    
    return df


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
    sensitivity = safe_div(tp, tp + fn)  # recall value
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
    scaler_path = Path("ml/models/scaler_v1.pkl")
    meta_path = Path("ml/models/isolation_forest_v1.meta.json")
    eval_data_path = Path("ml/data/eval.csv")
    
    # Load the model and scaler
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load scaler if it exists
    if scaler_path.exists():
        print(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        print("⚠️ No scaler found, predictions may be inaccurate")
        scaler = None
    
    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    contamination = meta.get("contamination", 0.031)
    expected_features = meta.get("expected_features", [])
    
    print(f"Expected features: {len(expected_features)}")
    print(f"Contamination rate: {contamination}")
    
    # Load evaluation data
    print(f"Loading evaluation data from {eval_data_path}")
    df_eval = pd.read_csv(eval_data_path)
    print(f"Loaded {len(df_eval)} samples")
    
    # Create engineered features if needed
    print("Creating/verifying engineered features...")
    df_eval = create_engineered_features(df_eval)
    
    # Separate features and labels
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
        raise ValueError("eval.csv must contain 'is_fraud', 'label', or 'Class' column")
    
    print(f"Using label column: {label_col}")
    print(f"Fraud cases in eval: {y_true.sum()}")
    
    # Ensure all expected features exist
    missing_features = []
    for feat in expected_features:
        if feat not in df_eval.columns:
            missing_features.append(feat)
            df_eval[feat] = 0.0  # Fill missing with 0
    
    if missing_features:
        print(f"⚠️ Missing {len(missing_features)} features, filled with 0:")
        print(f"   {missing_features[:5]}..." if len(missing_features) > 5 else f"   {missing_features}")
    
    # Extract features
    X_eval = df_eval[expected_features].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    
    # Scale features if scaler exists
    if scaler is not None:
        print("Scaling features...")
        X_eval = scaler.transform(X_eval)
    
    # Generate predictions
    print("Generating predictions...")
    anomaly_scores = model.decision_function(X_eval)
    anomaly_scores_normalized = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
    
    y_pred = decide_predictions(anomaly_scores_normalized, contamination, mode="quantile")
    
    # Calculate core metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    anomaly_rate = y_pred.sum() / len(y_pred)
    
    print(f"\n{'='*60}")
    print(f"PRELIMINARY RESULTS")
    print(f"{'='*60}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"{'='*60}\n")
    
    # Calculate extended metrics
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
    
    # Metrics for Decision Gate output
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
    
    # Full metrics for artifacts files
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
    
    # Save results for decision gate
    with open("ml_out/ml_metrics.json", "w") as f:
        json.dump(ml_metrics, f, indent=2)
    
    # Save full metrics 
    with open("ml_out/ml_full_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    
    # Save predictions to CSV 
    predictions_df = pd.DataFrame({
        "anomaly_score": anomaly_scores_normalized,
        "prediction": y_pred,
        "ground_truth": y_true
    })
    predictions_df.to_csv("ml_out/ml_predictions.csv", index=False)
    
    # Print summary results 
    print("\n" + "="*60)
    print("ML EVALUATION COMPLETE")
    print("="*60)
    print(f"Simplified metrics → ml_out/ml_metrics.json")
    print(f"Full metrics → ml_out/ml_full_metrics.json")
    print(f"Predictions → ml_out/ml_predictions.csv")
    print("="*60)
    print("\n### Simplified ML Metrics (for Decision Gate)")
    print(json.dumps(ml_metrics, indent=2))
    print("\n")


if __name__ == "__main__":
    main()
