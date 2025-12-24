# validation/summarize_trivy.py
"""
Summarize all validation/trivy_runs/**/trivy_metrics.json into CSV:

- validation/summary/trivy_runs_detailed.csv
- validation/summary/trivy_summary.csv   (mean ± std per profile)
"""

from __future__ import annotations

import csv
import json
import statistics as stats
from pathlib import Path


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def mean_std(vals):
    if not vals:
        return (0.0, 0.0)
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (stats.mean(vals), stats.stdev(vals))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = repo_root / "validation" / "trivy_runs"
    summary_dir = repo_root / "validation" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = sorted(runs_root.glob("**/trivy_metrics.json"))
    if not metrics_files:
        raise SystemExit(f"ERROR: No trivy_metrics.json found under {runs_root}")

    detailed_rows = []

    for mf in metrics_files:
        # Expected: validation/trivy_runs/<profile>/run_XX/trivy_metrics.json
        parts = mf.parts
        try:
            idx = parts.index("trivy_runs")
            profile = parts[idx + 1]
            run_name = parts[idx + 2]
        except Exception:
            profile = "unknown"
            run_name = mf.parent.name

        data = json.loads(mf.read_text(encoding="utf-8"))
        detailed_rows.append({
            "profile": profile,
            "run": run_name,
            "risk": data.get("risk", ""),
            "critical": int(data.get("critical", 0)),
            "TP": int(data.get("TP", 0)),
            "FP": int(data.get("FP", 0)),
            "FN": int(data.get("FN", 0)),
            "TN": int(data.get("TN", 0)),
            "precision": safe_float(data.get("precision")),
            "recall": safe_float(data.get("recall")),
            "f1": safe_float(data.get("f1")),
            "metrics_path": str(mf.relative_to(repo_root)),
        })

    # Detailed CSV
    detailed_csv = summary_dir / "trivy_runs_detailed.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        w.writeheader()
        w.writerows(detailed_rows)

    # Summary by profile
    by_profile = {}
    for r in detailed_rows:
        by_profile.setdefault(r["profile"], []).append(r)

    summary_rows = []
    for profile, rows in sorted(by_profile.items()):
        prec = [r["precision"] for r in rows]
        rec = [r["recall"] for r in rows]
        f1 = [r["f1"] for r in rows]
        crit = [r["critical"] for r in rows]

        mp, sp = mean_std(prec)
        mr, sr = mean_std(rec)
        mf, sf = mean_std(f1)
        mc, sc = mean_std(crit)

        risk_counts = {}
        for r in rows:
            risk_counts[r["risk"]] = risk_counts.get(r["risk"], 0) + 1

        summary_rows.append({
            "profile": profile,
            "n_runs": len(rows),
            "precision_mean": round(mp, 6),
            "precision_std": round(sp, 6),
            "recall_mean": round(mr, 6),
            "recall_std": round(sr, 6),
            "f1_mean": round(mf, 6),
            "f1_std": round(sf, 6),
            "critical_mean": round(mc, 6),
            "critical_std": round(sc, 6),
            "risk_counts": json.dumps(risk_counts, ensure_ascii=False),
        })

    summary_csv = summary_dir / "trivy_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"✓ Wrote {detailed_csv}")
    print(f"✓ Wrote {summary_csv}")


if __name__ == "__main__":
    main()
