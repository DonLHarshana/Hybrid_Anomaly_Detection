#!/usr/bin/env python3
"""
validation/run_ml_cv_trials.py

K-Fold Cross-Validation for Isolation Forest (unsupervised) using labels only for evaluation.

- Loads eval.csv (must contain label column: is_fraud OR label OR Class)
- Uses expected features from ml/models/isolation_forest_v1.meta.json if available
- For each seed:
    - StratifiedKFold split
    - Train IsolationForest ONLY on normal samples (y==0) in training fold
    - Evaluate on validation fold
- Writes per-fold detailed CSV: validation/ml_runs/ml_runs_detailed.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix
)

DEFAULT_META = Path("ml/models/isolation_forest_v1.meta.json")
DEFAULT_DATA = Path("ml/data/eval.csv")


def is_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:2]
        return any("git-lfs.github.com/spec" in line for line in head)
    except Exception:
        return False


def detect_label_column(df: pd.DataFrame) -> str:
    for col in ("is_fraud", "label", "Class"):
        if col in df.columns:
            return col
    raise ValueError("No label column found. Expected one of: is_fraud, label, Class")


def load_expected_features(meta_path: Path, df: pd.DataFrame, label_col: str) -> List[str]:
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            feats = meta.get("expected_features") or meta.get("FEATURE_COLS") or []
            feats = [f for f in feats if f in df.columns]
            if feats:
                return feats
        except Exception:
            pass

    # Fallback: all numeric columns except label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c != label_col]
    if not feats:
        raise ValueError("No usable feature columns found.")
    return feats


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    # AUC fails if only one class in y_true
    try:
        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.0


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(average_precision_score(y_true, scores))
    except Exception:
        return 0.0


def parse_seeds(s: str) -> List[int]:
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [42]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="Path to eval.csv (with labels)")
    ap.add_argument("--meta", default=str(DEFAULT_META), help="Path to meta json with expected features")
    ap.add_argument("--folds", type=int, default=5, help="K folds (default 5)")
    ap.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds for repeated CV")
    ap.add_argument("--contam", type=float, default=0.05, help="IsolationForest contamination")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--max-rows", type=int, default=None, help="Optional: limit rows for faster runs")
    ap.add_argument("--outdir", default="validation/ml_runs", help="Output folder")
    args = ap.parse_args()

    data_path = Path(args.data)
    meta_path = Path(args.meta)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if is_lfs_pointer(data_path):
        raise RuntimeError(
            f"{data_path} looks like a Git LFS pointer file.\n"
            f"Run: git lfs pull\n"
            f"Then re-run this validation."
        )

    df = pd.read_csv(data_path, nrows=args.max_rows)
    label_col = detect_label_column(df)
    y = df[label_col].astype(int).values

    features = load_expected_features(meta_path, df, label_col)
    X = df[features].copy()

    # Fill NA to avoid crashes
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    seeds = parse_seeds(args.seeds)
    folds = int(args.folds)

    rows = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx].values
            y_train = y[train_idx]

            X_val = X.iloc[val_idx].values
            y_val = y[val_idx]

            # Train ONLY on normal data (y==0)
            normal_mask = (y_train == 0)
            X_train_normal = X_train[normal_mask]

            # Scale using only training-normal
            scaler = StandardScaler()
            X_train_normal_s = scaler.fit_transform(X_train_normal)
            X_val_s = scaler.transform(X_val)

            model = IsolationForest(
                n_estimators=args.n_estimators,
                max_samples=args.max_samples,
                contamination=args.contam,
                random_state=seed,
                n_jobs=-1
            )
            model.fit(X_train_normal_s)

            # Predict: sklearn returns 1 normal, -1 anomaly
            pred = model.predict(X_val_s)
            y_hat = (pred == -1).astype(int)  # 1 = anomaly/fraud

            # Scores for AUC/AP: higher = more anomalous
            scores = -model.score_samples(X_val_s)

            prec = precision_score(y_val, y_hat, zero_division=0)
            rec = recall_score(y_val, y_hat, zero_division=0)
            f1 = f1_score(y_val, y_hat, zero_division=0)

            auc = safe_auc(y_val, scores)
            apv = safe_ap(y_val, scores)
            mcc = float(matthews_corrcoef(y_val, y_hat)) if len(np.unique(y_val)) > 1 else 0.0

            tn, fp, fn, tp = confusion_matrix(y_val, y_hat, labels=[0, 1]).ravel()

            rows.append({
                "seed": seed,
                "fold": fold_i,
                "folds": folds,
                "contamination": args.contam,
                "n_estimators": args.n_estimators,
                "max_samples": args.max_samples,
                "n_train_total": int(len(train_idx)),
                "n_train_normal": int(X_train_normal.shape[0]),
                "n_val": int(len(val_idx)),
                "val_anomalies": int(y_val.sum()),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "precision": float(round(prec, 6)),
                "recall": float(round(rec, 6)),
                "f1": float(round(f1, 6)),
                "roc_auc": float(round(auc, 6)),
                "avg_precision": float(round(apv, 6)),
                "mcc": float(round(mcc, 6)),
            })

    detailed_path = outdir / "ml_runs_detailed.csv"
    pd.DataFrame(rows).to_csv(detailed_path, index=False)
    print(f"✅ Wrote detailed results -> {detailed_path}")


if __name__ == "__main__":
    main()
