"""
Feature definitions for luxury cosmetics fraud detection.
Matches the luxury_cosmetics_fraud_analysis_2025.csv dataset.
"""

import pandas as pd

# These are the actual columns from your luxury cosmetics dataset
# After preprocessing, we use 11 numeric features
FEATURE_COLS = [
    "Customer_Age",
    "Customer_Loyalty_Tier",
    "Location",
    "Store_ID",
    "Product_SKU",
    "Product_Category",
    "Purchase_Amount",
    "Payment_Method",
    "Device_Type",
    "Fraud_Flag",
    "Footfall_Count"
]

def load_csv(path: str):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].astype(float)
    y = df["label"].astype(int) if "label" in df.columns else None
    return X, y, df
