# ml/src/generate_synthetic.py
from pathlib import Path
import numpy as np, pandas as pd
from utils import ensure_dir, log
from features import FEATURE_COLS

OUT_DIR = ensure_dir("ml/data")

def _rng(seed): return np.random.default_rng(int(seed))
def _clip01(x): return np.clip(x, 0.0, 1.0)

def synth_normal(n, rng):
    df = pd.DataFrame(index=range(n))
    # Example heuristics; adapt to your domain as needed:
    cols = set(FEATURE_COLS)
    def add_if(name, values): 
        if name in cols: df[name] = values

    add_if("amount",               np.abs(rng.normal(75, 50, n)))
    add_if("merchant_risk",        _clip01(rng.normal(0.3, 0.15, n)))
    add_if("velocity",             np.clip(rng.poisson(2, n), 0, 20))
    add_if("card_age_days",        np.clip(rng.normal(365, 200, n), 0, 4000))
    add_if("country_mismatch",     rng.choice([0,1], p=[0.97,0.03], size=n))
    add_if("ip_reputation",        _clip01(rng.normal(0.7, 0.15, n)))
    add_if("device_trust",         _clip01(rng.normal(0.8, 0.12, n)))
    add_if("txn_hour",             rng.integers(0,24,size=n))
    add_if("txn_dayofweek",        rng.integers(0,7,size=n))
    add_if("mcc_risk",             _clip01(rng.normal(0.3, 0.2, n)))
    add_if("bin_country_mismatch", rng.choice([0,1], p=[0.98,0.02], size=n))
    add_if("email_age_days",       np.clip(rng.normal(720, 400, n), 0, 5000))
    add_if("ip_proxy",             rng.choice([0,1], p=[0.97,0.03], size=n))
    add_if("shipping_distance_km", np.abs(rng.normal(50, 60, n)))
    add_if("past_chargebacks",     np.clip(rng.poisson(0.1, n), 0, 5))

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[FEATURE_COLS]

def synth_anomaly(n, rng):
    df = pd.DataFrame(index=range(n))
    cols = set(FEATURE_COLS)
    def add_if(name, values): 
        if name in cols: df[name] = values

    add_if("amount",               np.abs(rng.normal(1200, 700, n)))
    add_if("merchant_risk",        _clip01(rng.normal(0.8, 0.15, n)))
    add_if("velocity",             np.clip(rng.poisson(10, n), 0, 50))
    add_if("card_age_days",        np.clip(rng.normal(20, 30, n), 0, 4000))
    add_if("country_mismatch",     rng.choice([0,1], p=[0.6,0.4], size=n))
    add_if("ip_reputation",        _clip01(rng.normal(0.2, 0.15, n)))
    add_if("device_trust",         _clip01(rng.normal(0.3, 0.2, n)))
    add_if("txn_hour",             rng.choice([0,1,2,3,23], size=n))
    add_if("txn_dayofweek",        rng.integers(0,7,size=n))
    add_if("mcc_risk",             _clip01(rng.normal(0.8, 0.15, n)))
    add_if("bin_country_mismatch", rng.choice([0,1], p=[0.5,0.5], size=n))
    add_if("email_age_days",       np.clip(rng.normal(5, 10, n), 0, 5000))
    add_if("ip_proxy",             rng.choice([0,1], p=[0.4,0.6], size=n))
    add_if("shipping_distance_km", np.abs(rng.normal(1500, 800, n)))
    add_if("past_chargebacks",     np.clip(rng.poisson(2.5, n), 0, 20))

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[FEATURE_COLS]

def main(n_train=5000, n_eval=1000, anomaly_rate_eval=0.15, seed=42):
    rng = _rng(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = synth_normal(int(n_train), rng)
    (OUT_DIR / "train.csv").write_text(train.to_csv(index=False))
    log(f"Wrote ml/data/train.csv with {len(train)} rows")

    n_anom = int(n_eval * anomaly_rate_eval); n_norm = n_eval - n_anom
    eval_df = pd.concat([synth_normal(n_norm, rng), synth_anomaly(n_anom, rng)], ignore_index=True)
    labels = np.array([0]*n_norm + [1]*n_anom)
    idx = rng.permutation(n_eval)
    eval_df = eval_df.iloc[idx].reset_index(drop=True)
    eval_df["label"] = labels[idx].astype(int)
    (OUT_DIR / "eval.csv").write_text(eval_df.to_csv(index=False))
    log(f"Wrote ml/data/eval.csv with {len(eval_df)} rows (anomaly_rate≈{anomaly_rate_eval})")

if __name__ == "__main__":
    import os
    main(
        n_train=int(os.getenv("SYN_N_TRAIN", "5000")),
        n_eval=int(os.getenv("SYN_N_EVAL", "1000")),
        anomaly_rate_eval=float(os.getenv("SYN_EVAL_AR", "0.15")),
        seed=int(os.getenv("SYN_SEED", "42")),
    )
