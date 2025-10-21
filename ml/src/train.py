# ml/src/train.py
from pathlib import Path
import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin
from utils import ensure_dir, log, write_json
from features import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, DATE_TIME, ID_OR_DROP, FEATURE_COLS
)

MODEL_DIR  = ensure_dir("ml/models")
MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")
META_PATH  = Path("ml/models/isolation_forest_v1.meta.json")
TRAIN_CSV  = Path("ml/data/train.csv")

LABEL_NAMES = {"label","is_fraud","fraud","Fraud_Flag"}

# --------------------------
# Feature engineering pieces
# --------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time and cyclic features from Transaction_Date/Time.
    """
    out = df.copy()
    dt = pd.to_datetime(out.get("Transaction_Date"), errors="coerce")
    tt = pd.to_datetime(out.get("Transaction_Time"), format="%H:%M:%S", errors="coerce")

    out["txn_hour"]    = tt.dt.hour.fillna(0).astype(int)
    out["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)  # 0=Mon
    out["is_weekend"]  = out["day_of_week"].isin([5, 6]).astype(int)

    # Cyclic encodings help periodic signals
    out["hour_sin"] = np.sin(2*np.pi*out["txn_hour"]/24.0)
    out["hour_cos"] = np.cos(2*np.pi*out["txn_hour"]/24.0)
    out["dow_sin"]  = np.sin(2*np.pi*out["day_of_week"]/7.0)
    out["dow_cos"]  = np.cos(2*np.pi*out["day_of_week"]/7.0)
    return out

class GroupZScore(BaseEstimator, TransformerMixin):
    """
    z-score of a numeric column relative to group key(s).
    Example: z(Purchase_Amount) by Customer_ID highlights spikes vs a customer's norm.
    """
    def __init__(self, value_col: str, by: list[str], out_col: str, eps: float=1e-6):
        self.value_col = value_col
        self.by = by
        self.out_col = out_col
        self.eps = eps
        self._stats = None

    def fit(self, X: pd.DataFrame, y=None):
        g = X.groupby(self.by, dropna=False)[self.value_col]
        stats = g.agg(["mean", "std"]).rename(columns={"mean":"_mean","std":"_std"})
        stats["_std"] = stats["_std"].replace(0, self.eps).fillna(self.eps)
        self._stats = stats
        return self

    def transform(self, X: pd.DataFrame):
        out = X.copy()
        out = out.join(self._stats, on=self.by)
        z = (out[self.value_col] - out["_mean"]) / out["_std"]
        out[self.out_col] = z.replace([np.inf, -np.inf], 0).fillna(0)
        return out.drop(columns=["_mean", "_std"])

class RarityEncoder(BaseEstimator, TransformerMixin):
    """
    Adds rarity features for categorical cols:
      rarity_col = -log( freq(col=value) ), clipped at small floor.
    Rare devices/locations/categories often correlate with fraud risk.
    """
    def __init__(self, cols: list[str], min_count: int = 1):
        self.cols = cols
        self.min_count = min_count
        self._maps: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        n = max(1, len(X))
        for c in self.cols:
            vc = X[c].value_counts(dropna=False)
            vc = vc[vc >= self.min_count]
            freq = (vc / n).clip(lower=1e-9)
            self._maps[c] = (-np.log(freq)).to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        out = X.copy()
        for c in self.cols:
            m = self._maps.get(c, {})
            out[f"rarity_{c}"] = out[c].map(m).fillna(0.0)
        return out

def _safe_ohe(**kw):
    """
    Create OneHotEncoder with frequency cutoff when available.
    Falls back if sklearn version doesn't support `min_frequency`.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            **kw
        )
    except TypeError:
        # Older sklearn: no min_frequency
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

# -------------
# Train pipeline
# -------------
def main():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"{TRAIN_CSV} not found")

    log(f"Loading {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)

    # Safety: ensure label is not in features
    assert not (LABEL_NAMES & set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)), \
        f"Label leaked into features: {LABEL_NAMES & set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)}"

    # Ensure referenced cols exist to avoid crashes in FE steps
    for c in set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + DATE_TIME + ID_OR_DROP):
        if c not in df.columns:
            df[c] = np.nan

    # ---- Feature engineering (stateful) BEFORE encoding ----
    fe_steps = []
    fe_steps.append(("time", FunctionTransformer(add_time_features, validate=False)))

    # Amount z-scores by entity (only if keys exist)
    if "Purchase_Amount" in df.columns and "Customer_ID" in df.columns:
        fe_steps.append(("z_amt_by_customer",
                         GroupZScore("Purchase_Amount", ["Customer_ID"], "amt_z_by_customer")))
    if "Purchase_Amount" in df.columns and "Store_ID" in df.columns:
        fe_steps.append(("z_amt_by_store",
                         GroupZScore("Purchase_Amount", ["Store_ID"], "amt_z_by_store")))

    # Rarity encodings for available categoricals
    rarity_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if rarity_cols:
        fe_steps.append(("rarity", RarityEncoder(rarity_cols)))

    fe_pipeline = Pipeline(fe_steps)

    # Numeric after FE (scaling set)
    numeric_after = list(NUMERIC_FEATURES) + [
        "txn_hour","day_of_week","is_weekend","hour_sin","hour_cos","dow_sin","dow_cos",
        "amt_z_by_customer","amt_z_by_store"
    ]
    # Keep only those present (some FE may be skipped)
    numeric_after = [c for c in numeric_after if c in fe_pipeline.fit(df).transform(df).columns]

    # Categorical to one-hot (FE rarity_* are numeric already)
    categorical_after = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # Preprocess
    # Try frequency cutoff to reduce high-cardinality noise; fallback if unsupported
    try:
        ohe = _safe_ohe(min_frequency=0.01)
    except Exception:
        ohe = _safe_ohe()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_after),
            ("cat", ohe, categorical_after),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Isolation Forest with stronger, more stable params
    contam = float(os.getenv("IF_CONTAMINATION", "0.10"))  # tune to TRAIN fraud share if known
    trees  = int(os.getenv("IF_TREES", "400"))
    seed   = int(os.getenv("IF_SEED", "42"))

    model = IsolationForest(
        n_estimators=trees,
        contamination=contam,
        max_samples="auto",
        bootstrap=True,
        random_state=seed,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("fe",  fe_pipeline),
        ("pre", pre),
        ("iso", model),
    ])

    log("Fitting pipeline")
    pipe.fit(df)  # raw DF in; pipeline handles all transforms

    joblib.dump(pipe, MODEL_PATH)

    # Persist the post-encoding feature names (what the model actually sees)
    try:
        used_features = list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        used_features = []

    write_json(META_PATH, {
        "raw_feature_spec": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "engineered_time": ["txn_hour", "day_of_week", "is_weekend", "hour_sin","hour_cos","dow_sin","dow_cos"],
            "rarity_from": rarity_cols,
            "zscore_from": ["Customer_ID", "Store_ID"],
            "dropped": list(ID_OR_DROP) + list(DATE_TIME)
        },
        "expected_features": used_features,
        "model": "IsolationForest",
        "preprocessing": [
            "time_features", "GroupZScore(amount by customer/store)", "RarityEncoder(cat)",
            "StandardScaler(num)", "OneHot(cat)"
        ],
        "params": {"n_estimators": trees, "contamination": contam, "bootstrap": True, "random_state": seed},
        "version": "v2"
    })

    log(f"Saved model -> {MODEL_PATH}")
    log(f"Saved meta  -> {META_PATH}")

if __name__ == "__main__":
    main()
