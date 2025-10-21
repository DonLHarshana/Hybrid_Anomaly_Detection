"""
Edit FEATURE_COLS to match your dataset. Keep eval/train CSVs small for CI.
If 'label' column exists (1=anomaly, 0=normal), metrics are computed; otherwise they’re skipped.
"""
import pandas as pd

# Example features — replace with your real columns
NUMERIC_FEATURES = [
    "Customer_Age",
    "Purchase_Amount",
    "Footfall_Count",
]

CATEGORICAL_FEATURES = [
    "Customer_Loyalty_Tier",
    "Location",
    "Product_Category",
    "Payment_Method",
    "Device_Type",
]

DATE_TIME = ["Transaction_Date", "Transaction_Time"]

# Columns we intentionally drop / never feed into the model
# (IDs, PII, and the label)
ID_OR_DROP = [
    "Transaction_ID",
    "Customer_ID",
    "Store_ID",
    "Product_SKU",
    "IP_Address",
    "Fraud_Flag",     # <-- label; used only for evaluation
]



def load_csv(path: str):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].astype(float)
    y = df["label"].astype(int) if "label" in df.columns else None
    return X, y, df
