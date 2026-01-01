# ml/src/train.py
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils import ensure_dir, log, write_json
from features import FEATURE_COLS, build_feature_matrix

MODEL_DIR = ensure_dir("ml/models")
MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")
SCALER_PATH = Path("ml/models/scaler_v1.pkl")
META_PATH = Path("ml/models/isolation_forest_v1.meta.json")
TRAIN_CSV = Path("ml/data/train.csv")
METRICS_PATH = Path("ml/models/training_metrics.json")


def analyze_data(df):
    log("\n=== Dataset Analysis ===")
    log(f"Total samples: {len(df)}")

    if "is_fraud" in df.columns:
        label_col = "is_fraud"
    elif "label" in df.columns:
        label_col = "label"
    elif "Class" in df.columns:
        label_col = "Class"
    else:
        label_col = None

    if label_col:
        anomaly_rate = float(df[label_col].mean())
        log(f"Label col: {label_col}")
        log(f"Anomaly rate: {anomaly_rate:.6f} ({anomaly_rate*100:.3f}%)")
        return anomaly_rate, label_col

    log("No label column found - unsupervised training only")
    return None, None


def find_best_threshold(y_true, scores_norm):
    thresholds = np.percentile(scores_norm, np.linspace(50, 99.9, 200))
    best = {"f1": 0.0, "thr": float(thresholds[0]), "pred": None}
    for thr in thresholds:
        pred = (scores_norm >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "thr": float(thr), "pred": pred}
    return best["pred"], best["thr"], best["f1"]


def main(contamination=0.005, random_state=42, n_estimators=400, max_samples=512, evaluate=True):
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"{TRAIN_CSV} not found")

    log(f"Loading {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)

    anomaly_rate, label_col = analyze_data(df)

    # Build consistent features
    X = build_feature_matrix(df)

    metrics = {}
    best_threshold = None

    if evaluate and label_col is not None:
        y = df[label_col].astype(int).values

        log("Splitting data (70% train, 30% val)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=random_state, stratify=y
        )

        # Train ONLY on normal samples
        X_train_normal = X_train[y_train == 0]
        log(f"Training on normal-only: {len(X_train_normal)} samples")

        # Fit scaler ONLY on normal training data (prevents leakage)
        log("Fitting scaler on normal-only training data...")
        scaler = StandardScaler()
        X_train_normal_scaled = scaler.fit_transform(X_train_normal)
        X_val_scaled = scaler.transform(X_val)

        train_data = X_train_normal_scaled

    else:
        log("No labels or evaluate disabled: training on all rows (unsupervised-only)")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        train_data = X_scaled
        X_val_scaled = None
        y_val = None

    # Train Isolation Forest
    log("\nTraining IsolationForest with:")
    log(f"  n_estimators   : {n_estimators}")
    log(f"  contamination  : {contamination}")
    log(f"  max_samples    : {max_samples}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_data)
    log("✓ Model training completed")

    # Optional validation (score-based threshold like your evaluate.py)
    if X_val_scaled is not None and y_val is not None:
        log("\n=== Validation (score-based threshold) ===")
        raw_scores = model.decision_function(X_val_scaled)
        scores = -raw_scores
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        y_pred, thr, best_f1 = find_best_threshold(y_val, scores_norm)
        best_threshold = thr

        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "best_threshold": float(thr),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        }

        log(f"Best threshold: {thr:.4f}")
        log(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        write_json(METRICS_PATH, metrics)
        log(f"✓ Metrics saved to {METRICS_PATH}")

    # Save model + scaler + meta
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "expected_features": FEATURE_COLS,
        "contamination": float(contamination),
        "n_estimators": int(n_estimators),
        "max_samples": int(max_samples),
        "random_state": int(random_state),
        "best_threshold": float(best_threshold) if best_threshold is not None else None,
        "metrics": metrics,
    }
    write_json(META_PATH, meta)

    log(f"\n✓ Model saved  -> {MODEL_PATH}")
    log(f"✓ Scaler saved -> {SCALER_PATH}")
    log(f"✓ Meta saved   -> {META_PATH}")
    return model, scaler, metrics


if __name__ == "__main__":
    import os

    contamination = float(os.getenv("IF_CONTAMINATION", "0.005"))
    random_state = int(os.getenv("IF_SEED", "42"))
    n_estimators = int(os.getenv("IF_TREES", "400"))
    max_samples = int(os.getenv("IF_MAX_SAMPLES", "512"))
    evaluate = os.getenv("IF_EVALUATE", "true").lower() == "true"

    main(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        max_samples=max_samples,
        evaluate=evaluate,
    )
