#!/usr/bin/env python3
"""
validation/summarize_lof.py

Summarizes LOF K-fold validation runs:
- Reads: validation/lof_runs/lof_runs_detailed.csv
- Outputs:
  - validation/summary/lof_summary.csv (mean/std per metric across ALL folds+seeds)
  - validation/summary/lof_by_seed.csv (mean per seed)
"""

from pathlib import Path
import pandas as pd

IN_PATH = Path("validation/lof_runs/lof_runs_detailed.csv")
OUT_DIR = Path("validation/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["precision", "recall", "f1", "roc_auc", "avg_precision", "mcc"]

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    by_seed = df.groupby("seed")[METRICS].mean().reset_index()
    by_seed_path = OUT_DIR / "lof_by_seed.csv"
    by_seed.to_csv(by_seed_path, index=False)

    summary = {}
    for m in METRICS:
        summary[f"{m}_mean"] = float(df[m].mean())
        summary[f"{m}_std"]  = float(df[m].std(ddof=0))

    summary["total_rows"] = int(len(df))
    summary["seeds"] = int(df["seed"].nunique())
    summary["folds"] = int(df["folds"].iloc[0]) if "folds" in df.columns and len(df) else 0
    summary["contamination"] = float(df["contamination"].iloc[0]) if "contamination" in df.columns and len(df) else 0.0
    summary["n_neighbors"] = int(df["n_neighbors"].iloc[0]) if "n_neighbors" in df.columns and len(df) else 0

    out = pd.DataFrame([summary])
    summary_path = OUT_DIR / "lof_summary.csv"
    out.to_csv(summary_path, index=False)

    print(f"✅ Wrote per-seed summary -> {by_seed_path}")
    print(f"✅ Wrote overall summary  -> {summary_path}")

if __name__ == "__main__":
    main()
