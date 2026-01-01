"""
Feature definitions for Credit Card Fraud Detection
Dataset: Kaggle Credit Card Fraud
"""

import numpy as np
import pandas as pd

V_FEATURES = [f"V{i}" for i in range(1, 29)]

# Match your Colab engineered features
AMOUNT_FEATURES = ["Amount", "Amount_log", "Amount_squared"]
TIME_FEATURES = ["Time_sin", "Time_cos"]

# Extra useful summary stats from V1..V28
V_STAT_FEATURES = ["V_mean", "V_std", "V_max", "V_min"]

FEATURE_COLS = V_FEATURES + AMOUNT_FEATURES + TIME_FEATURES + V_STAT_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing engineered features consistently for BOTH train and eval."""
    df = df.copy()

    # --- Amount ---
    if "Amount" in df.columns:
        amt = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    elif "amount" in df.columns:
        amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["Amount"] = amt
    else:
        amt = pd.Series(0.0, index=df.index)
        df["Amount"] = amt

    if "Amount_log" not in df.columns:
        df["Amount_log"] = np.log1p(amt.clip(lower=0))

    if "Amount_squared" not in df.columns:
        df["Amount_squared"] = (amt.fillna(0.0) ** 2)

    # --- Time (raw seconds) → sin/cos ---
    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"], errors="coerce").fillna(0.0)
    elif "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").fillna(0.0)
        df["Time"] = t
    else:
        t = pd.Series(0.0, index=df.index)
        df["Time"] = t

    # If Time_sin/cos missing, create them
    if "Time_sin" not in df.columns or "Time_cos" not in df.columns:
        period = 24 * 3600.0
        df["Time_sin"] = np.sin(2 * np.pi * (t / period))
        df["Time_cos"] = np.cos(2 * np.pi * (t / period))

    # --- Ensure V features exist ---
    for c in V_FEATURES:
        if c not in df.columns:
            df[c] = 0.0

    vmat = df[V_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # --- V statistics ---
    df["V_mean"] = vmat.mean(axis=1)
    df["V_std"] = vmat.std(axis=1)
    df["V_max"] = vmat.max(axis=1)
    df["V_min"] = vmat.min(axis=1)

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature matrix with all FEATURE_COLS present."""
    df2 = add_derived_features(df)
    for c in FEATURE_COLS:
        if c not in df2.columns:
            df2[c] = 0.0
    X = df2[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X
