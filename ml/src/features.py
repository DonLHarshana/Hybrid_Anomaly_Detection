"""
Feature definitions for Credit Card Fraud Detection
Dataset: Kaggle Credit Card Fraud (100K transactions sample)
"""

import pandas as pd
import numpy as np

# V1-V28: PCA-transformed features (anonymized)
V_FEATURES = [f'V{i}' for i in range(1, 29)]

# Amount features
AMOUNT_FEATURES = ['Amount', 'Amount_log', 'Amount_sqrt', 'Amount_cube']

# Time features
TIME_FEATURES = ['Time_sin', 'Time_cos']

# V statistics
V_STAT_FEATURES = ['V_mean', 'V_std', 'V_max', 'V_min']

# Complete feature list (38 features)
FEATURE_COLS = V_FEATURES + AMOUNT_FEATURES + TIME_FEATURES + V_STAT_FEATURES

def load_csv(path: str):
    """
    Load CSV and separate features from labels
    
    Args:
        path: Path to CSV file
    
    Returns:
        tuple: (X, y, df) where X is features, y is labels, df is full dataframe
    """
    df = pd.read_csv(path)
    
    # Check available features
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    
    if not available_features:
        raise ValueError(f"No expected features found in {path}")
    
    X = df[available_features].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    # Get labels
    if "label" in df.columns:
        y = df["label"].astype(int)
    elif "Class" in df.columns:
        y = df["Class"].astype(int)
    else:
        y = None
    
    return X, y, df
