# ml/src/train.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils import ensure_dir, log, write_json
from features import FEATURE_COLS
import joblib

MODEL_DIR = ensure_dir("ml/models")
MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")
SCALER_PATH = Path("ml/models/scaler_v1.pkl")  # NEW: Save scaler
META_PATH = Path("ml/models/isolation_forest_v1.meta.json")
TRAIN_CSV = Path("ml/data/train.csv")
METRICS_PATH = Path("ml/models/training_metrics.json")  # NEW: Save metrics


def create_advanced_features(df):
    """
    Add derived features for better anomaly detection
    """
    log("Creating advanced features...")
    
    # Example: Add log-transformed amount if 'amount' exists
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'].fillna(0))
        df['amount_squared'] = df['amount'].fillna(0) ** 2
    
    # Example: Time-based features if 'timestamp' exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour.fillna(0)
        df['day_of_week'] = df['timestamp'].dt.dayofweek.fillna(0)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df


def analyze_data(df):
    """
    Analyze dataset characteristics
    """
    log("\n=== Dataset Analysis ===")
    log(f"Total samples: {len(df)}")
    
    if 'is_fraud' in df.columns or 'label' in df.columns:
        label_col = 'is_fraud' if 'is_fraud' in df.columns else 'label'
        anomaly_rate = df[label_col].mean()
        log(f"Anomaly rate: {anomaly_rate:.4f} ({anomaly_rate*100:.2f}%)")
        log(f"Normal samples: {(df[label_col]==0).sum()}")
        log(f"Anomaly samples: {(df[label_col]==1).sum()}")
        return anomaly_rate, label_col
    else:
        log("No label column found - assuming unsupervised training")
        return None, None


def main(contamination=0.05, random_state=42, n_estimators=200, 
         max_samples=256, evaluate=True):
    
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"{TRAIN_CSV} not found")

    log(f"Loading {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    
    # Analyze data
    anomaly_rate, label_col = analyze_data(df)
    
    # Create advanced features
    df = create_advanced_features(df)
    
    # Ensure all required features exist
    for c in FEATURE_COLS:
        if c not in df.columns:
            log(f"Warning: Feature '{c}' not found, filling with 0")
            df[c] = 0.0
    
    # Extract features
    Xdf = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    # NEW: Feature Scaling (CRITICAL for Isolation Forest)
    log("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xdf)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    
    log(f"Feature scaling completed (mean≈0, std≈1)")
    log(f"Scaled mean: {X_scaled.mean().mean():.4f}")
    log(f"Scaled std: {X_scaled.std().mean():.4f}")
    
    # Adjust contamination based on actual data if possible
    if anomaly_rate is not None and contamination != anomaly_rate:
        log(f"Note: contamination={contamination}, actual rate={anomaly_rate:.4f}")
        log(f"Consider setting contamination close to actual rate")
    
    # Split data for evaluation if labels exist
    if evaluate and label_col is not None:
        log("Splitting data for evaluation (70% train, 30% test)...")
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=random_state, stratify=y
        )
        
        # Train only on NORMAL samples (unsupervised approach)
        log(f"Training on {(y_train==0).sum()} normal samples only...")
        X_train_normal = X_train[y_train == 0]
        train_data = X_train_normal
    else:
        log("Training on all samples (no labels for evaluation)...")
        X_train, X_test, y_train, y_test = X_scaled, None, None, None
        train_data = X_scaled

    # Train Isolation Forest
    log(f"\nTraining IsolationForest with:")
    log(f"  - n_estimators: {n_estimators}")
    log(f"  - contamination: {contamination}")
    log(f"  - max_samples: {max_samples}")
    
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    model.fit(train_data)
    log("✓ Model training completed")

    # Evaluate model if test data exists
    metrics = {}
    if X_test is not None and y_test is not None:
        log("\n=== Model Evaluation ===")
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1/1 to 1/0
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "FPR": float(fpr),
            "FNR": float(fnr)
        }
        
        log(f"Precision: {precision:.4f}")
        log(f"Recall:    {recall:.4f}")
        log(f"F1 Score:  {f1:.4f}")
        log(f"FP Rate:   {fpr:.4f}")
        log(f"FN Rate:   {fnr:.4f}")
        log(f"\nConfusion Matrix:")
        log(f"  TP: {tp:4d}  |  FP: {fp:4d}")
        log(f"  FN: {fn:4d}  |  TN: {tn:4d}")
        
        # Save metrics
        write_json(METRICS_PATH, metrics)
        log(f"✓ Metrics saved to {METRICS_PATH}")

    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    meta = {
        "expected_features": FEATURE_COLS,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "random_state": random_state,
        "metrics": metrics
    }
    write_json(META_PATH, meta)

    log(f"\n✓ Model saved  -> {MODEL_PATH}")
    log(f"✓ Scaler saved -> {SCALER_PATH}")
    log(f"✓ Meta saved   -> {META_PATH}")
    
    return model, scaler, metrics


if __name__ == "__main__":
    import os
    
    # Get hyperparameters from environment or use defaults
    contamination = float(os.getenv("IF_CONTAMINATION", "0.05"))  # Changed default
    random_state = int(os.getenv("IF_SEED", "42"))
    n_estimators = int(os.getenv("IF_TREES", "200"))
    max_samples = int(os.getenv("IF_MAX_SAMPLES", "256"))
    evaluate = os.getenv("IF_EVALUATE", "true").lower() == "true"
    
    log(f"\n{'='*50}")
    log("Starting Isolation Forest Training")
    log(f"{'='*50}\n")
    
    model, scaler, metrics = main(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        max_samples=max_samples,
        evaluate=evaluate
    )
    
    log(f"\n{'='*50}")
    log("Training completed successfully!")
    log(f"{'='*50}\n")
