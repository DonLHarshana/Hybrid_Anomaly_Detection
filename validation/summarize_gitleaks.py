#!/usr/bin/env python3
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

def main():
    detailed = ROOT / "validation" / "gitleaks_runs" / "gitleaks_runs_detailed.csv"
    summary_dir = ROOT / "validation" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out = summary_dir / "gitleaks_summary.csv"

    rows = []
    with open(detailed, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # convert numeric fields
            for k in ["findings","TP","FP","FN"]:
                r[k] = int(r[k])
            for k in ["precision","recall","f1"]:
                r[k] = float(r[k])
            rows.append(r)

    profiles = sorted(set(r["profile"] for r in rows))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "profile","runs",
            "mean_precision","mean_recall","mean_f1",
            "min_f1","max_f1",
            "mean_findings"
        ])
        w.writeheader()

        for p in profiles:
            rr = [r for r in rows if r["profile"] == p]
            w.writerow({
                "profile": p,
                "runs": len(rr),
                "mean_precision": round(mean(r["precision"] for r in rr), 6),
                "mean_recall": round(mean(r["recall"] for r in rr), 6),
                "mean_f1": round(mean(r["f1"] for r in rr), 6),
                "min_f1": round(min(r["f1"] for r in rr), 6),
                "max_f1": round(max(r["f1"] for r in rr), 6),
                "mean_findings": round(mean(r["findings"] for r in rr), 3),
            })

    print(f"✅ Wrote {out}")

if __name__ == "__main__":
    main()
