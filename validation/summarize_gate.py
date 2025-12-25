#!/usr/bin/env python3
"""
validation/summarize_gate.py

Summarizes hybrid gate trials.
Input:  validation/gate_runs/gate_runs_detailed.csv
Output: validation/summary/gate_summary.csv
"""

from pathlib import Path
import pandas as pd

IN_PATH = Path("validation/gate_runs/gate_runs_detailed.csv")
OUT_DIR = Path("validation/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    # Decision counts
    def count_decisions(x: pd.Series):
        return pd.Series({
            "ACCEPT": int((x == "ACCEPT").sum()),
            "HOLD": int((x == "HOLD").sum()),
            "REJECT": int((x == "REJECT").sum()),
        })

    counts = df.groupby("profile")["gate_decision"].apply(count_decisions).reset_index()

    # Means (optional but useful)
    means = df.groupby("profile").agg({
        "high_severity_total": "mean",
        "critical": "mean",
        "high": "mean",
        "trivy_precision": "mean",
        "trivy_recall": "mean",
        "trivy_f1": "mean",
        "ml_precision": "mean",
        "ml_recall": "mean",
        "ml_f1": "mean",
    }).reset_index()

    out = counts.merge(means, on="profile", how="left")

    out_path = OUT_DIR / "gate_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Wrote: {out_path}")

if __name__ == "__main__":
    main()
