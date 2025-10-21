"""
Edit FEATURE_COLS to match your dataset. Keep eval/train CSVs small for CI.
If 'label' column exists (1=anomaly, 0=normal), metrics are computed; otherwise they’re skipped.
"""
import pandas as pd

# Example features — replace with your real columns
FEATURE_COLS = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Transaction_Time",
    "Customer_Age",
    "Customer_Loyalty_Tier",
    "Location",
    "Store_ID",
    "Product_SKU",
    "Product_Category",
    "Purchase_Amount",
    "Payment_Method",
    "Device_Type",
    "IP_Address",
    "Fraud_Flag",
    "Footfall_Count"
]

def load_csv(path: str):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].astype(float)
    y = df["label"].astype(int) if "label" in df.columns else None
    return X, y, df
