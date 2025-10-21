# ml/src/time_features.py
import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Parse timestamps
    dt = pd.to_datetime(out.get("Transaction_Date"), errors="coerce")
    tt = pd.to_datetime(out.get("Transaction_Time"), format="%H:%M:%S", errors="coerce")
    # Basic discrete time features
    out["txn_hour"]    = tt.dt.hour.fillna(0).astype(int)
    out["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)  # 0=Mon
    out["is_weekend"]  = out["day_of_week"].isin([5, 6]).astype(int)
    # Cyclic encodings (helps trees/isolation in periodic signals)
    out["hour_sin"] = np.sin(2*np.pi*out["txn_hour"]/24.0)
    out["hour_cos"] = np.cos(2*np.pi*out["txn_hour"]/24.0)
    out["dow_sin"]  = np.sin(2*np.pi*out["day_of_week"]/7.0)
    out["dow_cos"]  = np.cos(2*np.pi*out["day_of_week"]/7.0)
    return out
