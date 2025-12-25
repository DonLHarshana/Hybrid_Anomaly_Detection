#!/usr/bin/env python3
"""
validation/summarize_gate.py

Reads:
- validation/gate_runs/gate_runs_detailed.csv

Writes:
- validation/summary/gate_summary.csv
- validation/summary/gate_ablation_summary.csv

Purpose:
- gate_summary.csv: normal decision distribution + mean metrics (thesis results)
- gate_ablation_summary.csv: shows why hybrid is needed (Trivy-only vs ML-only vs Hybrid vs oracle)
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def main():
    runs_csv = Path("validation/gate_runs/gate_runs_detailed.csv")
    out_dir = Path("validation/summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_csv.exists():
        raise FileNotFoundError(f"Missing: {runs_csv}")

    df = pd.read_csv(runs_csv)

    # ----------------- gate_summary.csv -----------------
    # Decision counts per profile
    decision_counts = (
        df.pivot_table(index="profile", columns="gate_decision", values="run", aggfunc="count", fill_value=0)
        .rename(columns=lambda c: f"gate_{c.lower()}_count")
        .reset_index()
    )

    metrics_means = (
        df.groupby("profile", as_index=False)
          .agg(
              runs=("run", "count"),
              high_severity_mean=("high_severity_total", "mean"),
              trivy_precision_mean=("trivy_precision", "mean"),
              trivy_recall_mean=("trivy_recall", "mean"),
              trivy_f1_mean=("trivy_f1", "mean"),
              ml_anomaly_rate_mean=("ml_anomaly_rate", "mean"),
              ml_precision_mean=("ml_precision", "mean"),
              ml_recall_mean=("ml_recall", "mean"),
              ml_f1_mean=("ml_f1", "mean"),
          )
    )

    gate_summary = metrics_means.merge(decision_counts, on="profile", how="left").fillna(0)
    gate_summary_path = out_dir / "gate_summary.csv"
    gate_summary.to_csv(gate_summary_path, index=False)

    # ----------------- gate_ablation_summary.csv -----------------
    # Match rates against oracle by method
    ablation = (
        df.groupby("profile", as_index=False)
          .agg(
              runs=("run", "count"),
              oracle_decision=("oracle_decision", "first"),
              hybrid_match_rate=("oracle_match_hybrid", "mean"),
              trivy_only_match_rate=("oracle_match_trivy_only", "mean"),
              ml_only_match_rate=("oracle_match_ml_only", "mean"),
          )
    )

    # Also include how many times each method produced each decision (nice for thesis)
    def add_counts(method_col: str, prefix: str):
        c = (
            df.pivot_table(index="profile", columns=method_col, values="run", aggfunc="count", fill_value=0)
              .rename(columns=lambda x: f"{prefix}_{str(x).lower()}_count")
              .reset_index()
        )
        return c

    ablation = ablation.merge(add_counts("trivy_only_decision", "trivy_only"), on="profile", how="left")
    ablation = ablation.merge(add_counts("ml_only_decision", "ml_only"), on="profile", how="left")
    ablation = ablation.merge(add_counts("hybrid_decision", "hybrid"), on="profile", how="left")

    ablation_path = out_dir / "gate_ablation_summary.csv"
    ablation.to_csv(ablation_path, index=False)

    print(f"✅ Wrote: {gate_summary_path}")
    print(f"✅ Wrote: {ablation_path}")


if __name__ == "__main__":
    main()
