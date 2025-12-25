#!/usr/bin/env python3
"""
validation/run_gate_trials.py

Runs full hybrid pipeline multiple times per injection profile:
- generate payment set (INJECT_PROFILE)
- trivy scan -> score_trivy
- ML evaluate (Isolation Forest)
- decision_gate (final ACCEPT/HOLD/REJECT)
Outputs:
- validation/gate_runs/gate_runs_detailed.csv
- per-run JSON snapshots under validation/gate_runs/<profile>/run_XX/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List


def run(cmd: List[str], env: Dict[str, str] | None = None, allow_fail: bool = False) -> None:
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, env=env, text=True)
    if r.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_get(d: Dict[str, Any], k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--profiles", default="clean,low,medium,high")
    ap.add_argument("--payment-id", default="0001")
    ap.add_argument("--template", default="trivy/payment_set_template")
    ap.add_argument("--ml_contam", default="0.005")
    ap.add_argument("--ml_mode", default="bestf1")
    ap.add_argument("--outdir", default="validation/gate_runs")
    args = ap.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    detailed_csv = outdir / "gate_runs_detailed.csv"

    # Write header once
    fieldnames = [
        "profile", "run",
        "trivy_risk", "critical", "high", "medium", "low",
        "trivy_TP", "trivy_FP", "trivy_FN",
        "trivy_precision", "trivy_recall", "trivy_f1",
        "ml_precision", "ml_recall", "ml_f1", "ml_auc", "ml_pr_auc",
        "gate_decision", "gate_reason",
        "high_severity_total",
        "paths_trivy_metrics", "paths_ml_metrics", "paths_gate_out"
    ]

    # overwrite detailed CSV each time
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for profile in profiles:
            for i in range(1, args.runs + 1):
                run_name = f"run_{i:02d}"
                snap_dir = outdir / profile / run_name
                snap_dir.mkdir(parents=True, exist_ok=True)

                # clean working dirs each trial
                for p in ("datasets", "trivy_out", "ml_out", "artifacts"):
                    shutil.rmtree(p, ignore_errors=True)
                Path("datasets").mkdir(exist_ok=True)
                Path("trivy_out").mkdir(exist_ok=True)
                Path("ml_out").mkdir(exist_ok=True)
                Path("artifacts").mkdir(exist_ok=True)

                env = os.environ.copy()
                env["INJECT_PROFILE"] = profile
                env["ML_CONTAM"] = str(args.ml_contam)
                env["ML_DECISION_MODE"] = str(args.ml_mode)

                # 1) Generate payment set
                run([
                    "python", "trivy/make_payment_set_trivy.py",
                    "--id", args.payment_id,
                    "--template", args.template
                ], env=env)

                ds = Path(f"datasets/payment_set_{args.payment_id}")
                gt_csv = ds / "ground_truth" / "secrets.csv"

                # 2) Trivy scan (secrets)
                run([
                    "trivy", "fs",
                    "--scanners", "secret",
                    "--format", "json",
                    "-o", "trivy_out/scan.json",
                    str(ds)
                ], allow_fail=True)

                # 3) Score Trivy
                run([
                    "python", "trivy/score_trivy.py",
                    "--scan", "trivy_out/scan.json",
                    "--gt-csv", str(gt_csv),
                    "--out", "trivy_out/trivy_metrics.json"
                ])

                # 4) ML Evaluate
                run(["python", "ml/src/evaluate.py"], env=env)

                # 5) Decision Gate
                run(["python", "ml/src/decision_gate.py"], env=env)

                trivy_metrics = read_json(Path("trivy_out/trivy_metrics.json"))
                ml_metrics = read_json(Path("ml_out/ml_metrics.json"))
                gate_out = read_json(Path("ml_out/gate_out.json"))

                # Snapshot JSON outputs
                trivy_metrics_path = snap_dir / "trivy_metrics.json"
                ml_metrics_path = snap_dir / "ml_metrics.json"
                gate_out_path = snap_dir / "gate_out.json"

                trivy_metrics_path.write_text(json.dumps(trivy_metrics, indent=2), encoding="utf-8")
                ml_metrics_path.write_text(json.dumps(ml_metrics, indent=2), encoding="utf-8")
                gate_out_path.write_text(json.dumps(gate_out, indent=2), encoding="utf-8")

                critical = int(safe_get(trivy_metrics, "critical", 0) or 0)
                high = int(safe_get(trivy_metrics, "high", 0) or 0)
                medium = int(safe_get(trivy_metrics, "medium", 0) or 0)
                low = int(safe_get(trivy_metrics, "low", 0) or 0)
                high_sev_total = critical + high

                row = {
                    "profile": profile,
                    "run": run_name,
                    "trivy_risk": safe_get(trivy_metrics, "risk", ""),
                    "critical": critical,
                    "high": high,
                    "medium": medium,
                    "low": low,
                    "trivy_TP": safe_get(trivy_metrics, "TP", None),
                    "trivy_FP": safe_get(trivy_metrics, "FP", None),
                    "trivy_FN": safe_get(trivy_metrics, "FN", None),
                    "trivy_precision": safe_get(trivy_metrics, "precision", None),
                    "trivy_recall": safe_get(trivy_metrics, "recall", None),
                    "trivy_f1": safe_get(trivy_metrics, "f1", None),

                    # ML keys may vary; keep robust:
                    "ml_precision": safe_get(ml_metrics, "precision", safe_get(ml_metrics, "prec", None)),
                    "ml_recall": safe_get(ml_metrics, "recall", safe_get(ml_metrics, "rec", None)),
                    "ml_f1": safe_get(ml_metrics, "f1", None),
                    "ml_auc": safe_get(ml_metrics, "roc_auc", safe_get(ml_metrics, "auc", None)),
                    "ml_pr_auc": safe_get(ml_metrics, "avg_precision", safe_get(ml_metrics, "pr_auc", None)),

                    "gate_decision": safe_get(gate_out, "decision", ""),
                    "gate_reason": safe_get(gate_out, "reason", safe_get(gate_out, "explanation", "")),
                    "high_severity_total": high_sev_total,

                    "paths_trivy_metrics": str(trivy_metrics_path),
                    "paths_ml_metrics": str(ml_metrics_path),
                    "paths_gate_out": str(gate_out_path),
                }

                w.writerow(row)
                print(f"✅ {profile} {run_name} -> decision={row['gate_decision']} (high_sev={high_sev_total})")

    print(f"\n✅ Wrote: {detailed_csv}")


if __name__ == "__main__":
    main()
