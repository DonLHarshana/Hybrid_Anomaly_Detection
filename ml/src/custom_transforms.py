# ml/src/custom_transforms.py
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class GroupZScore(BaseEstimator, TransformerMixin):
    """
    Compute z-score of a numeric column relative to group key(s).
    Example: zscore of Purchase_Amount per Customer_ID (spikes flag).
    """
    def __init__(self, value_col: str, by: List[str], out_col: Optional[str]=None, eps: float=1e-6):
        self.value_col = value_col
        self.by = by
        self.out_col = out_col or f"{value_col}_z_by_{'_'.join(by)}"
        self.eps = eps
        self._stats = None

    def fit(self, X: pd.DataFrame, y=None):
        g = X.groupby(self.by)[self.value_col]
        self._stats = g.agg(["mean", "std"]).rename(columns={"mean":"_mean","std":"_std"})
        self._stats["_std"] = self._stats["_std"].replace(0, self.eps).fillna(self.eps)
        return self

    def transform(self, X: pd.DataFrame):
        out = X.copy()
        out = out.join(self._stats, on=self.by)
        z = (out[self.value_col] - out["_mean"]) / out["_std"]
        out[self.out_col] = z.replace([np.inf, -np.inf], 0).fillna(0)
        return out.drop(columns=["_mean","_std"])

class RarityEncoder(BaseEstimator, TransformerMixin):
    """
    For each categorical column, add numeric feature rarity_col = -log(freq(col=value)).
    Captures rare categories/devices/locations that often correlate with fraud.
    """
    def __init__(self, cols: List[str], min_count: int = 1):
        self.cols = cols
        self.min_count = min_count
        self._maps = {}

    def fit(self, X: pd.DataFrame, y=None):
        n = len(X)
        for c in self.cols:
            counts = X[c].value_counts()
            counts = counts[counts >= self.min_count]
            freq = (counts / n).clip(lower=1e-9)
            self._maps[c] = (-np.log(freq)).to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        out = X.copy()
        for c in self.cols:
            m = self._maps.get(c, {})
            out[f"rarity_{c}"] = out[c].map(m).fillna(0.0)
        return out
