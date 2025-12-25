#!/usr/bin/env python3
"""
validation/run_gate_trials.py

Full HYBRID validation (Trivy + ML + Decision Gate) with ABLATION baselines:
- Oracle decision (expected outcome by profile)
- Trivy-only decision (STRICT baseline)
- ML-only decision (simple baseline)
- Hybrid decision (your decision_gate.py)

Why:
To prove hybrid is needed: Trivy-only and ML-only each fail in some profiles, hybrid matches oracle.

Outputs:
- validation/gate_runs/gate_runs_detailed.csv
- Per-run snapshots under validation/gate_runs/<profile>/run_XX/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional


def run(cmd: List[str], env: Optional[Dict[str, str]] = None, allow_fail: bool = False, capture: bool = False) -> subprocess.CompletedProcess:
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, env=env, text=True, capture_output=capture)
    if r.returncode != 0 and not allow_fail:
        if r.stderr:
            print("STDERR:\n", r.stderr)
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}")
    return r


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_get(d: Dict[str, Any], k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


# ---------- ORACLE + BASELINES (for thesis proof) ----------

def oracle_decision(profile: str) -> str:
    """
    Expected "ground truth" gate decision by controlled injection profile.
    This is your validation target for the hybrid decision.
    """
    m = {
        "clean": "ACCEPT",
        "low": "HOLD",
        "medium": "HOLD",
        "high": "REJECT",
    }
    return m.get(profile, "HOLD")


def trivy_only_strict_decision(critical: int, high: int, medium: int, low: int) -> str:
    """
    Typical STRICT rule-based gate used in many CI setups:
    - any CRITICAL/HIGH => REJECT (blocks pipeline)
    - else any MEDIUM/LOW => HOLD (manual review)
    - else => ACCEPT
    This baseline is intentionally strict; it helps show hybrid reduces over-blocking.
    """
    if (critical + high) > 0:
        return "REJECT"
    if (medium + low) > 0:
        return "HOLD"
    return "ACCEPT"


def ml_only_decision(anomaly_rate: float, f1: float, ar_thresh: float, f1_thresh: float) -> str:
    """
    Simple ML-only baseline:
    - if anomaly_rate is high OR f1 too low => HOLD
    - else => ACCEPT
    With your current eval metrics (anomaly_rate ~0.001, f1 ~0.268), this becomes ACCEPT,
    which demonstrates ML-only cannot detect secret leakage profiles.
    """
    if (anomaly_rate >= ar_thresh) or (f1 < f1_thresh):
        return "HOLD"
    return "ACCEPT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--profiles", default="clean,low,medium,high")
    ap.add_argument("--payment-id", default="0001")
    ap.add_argument("--template", default="trivy/payment_set_template")
    ap.add_argument("--ml_contam", default="0.005")
    ap.add_argument("--ml_mode", default="bestf1")
    ap.add_argument("--outdir", default="validation/gate_runs")

    # ML-only baseline thresholds
    ap.add_argument("--ml_only_ar_thresh", type=float, default=0.01, help="ML-only HOLD if anomaly_rate >= this")
    ap.add_argument("--ml_only_f1_thresh", type=float, default=0.15, help="ML-only HOLD if F1 < this")

    args = ap.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    detailed_csv = outdir / "gate_runs_detailed.csv"

    fieldnames = [
        "profile", "run",

        # Oracle + baselines
        "oracle_decision",
        "trivy_only_decision",
        "ml_only_decision",
        "hybrid_decision",

        "oracle_match_trivy_only",
        "oracle_match_ml_only",
        "oracle_match_hybrid",

        # Trivy metrics
        "trivy_risk", "critical", "high", "medium", "low",
        "trivy_TP", "trivy_FP", "trivy_FN",
        "trivy_precision", "trivy_recall", "trivy_f1",
        "high_severity_total",

        # ML metrics
        "ml_anomaly_rate",
        "ml_precision", "ml_recall", "ml_f1", "ml_auc", "ml_pr_auc",

        # Gate output
        "gate_decision", "gate_reason", "gate_exit_code",

        # Snapshot paths
        "paths_trivy_metrics", "paths_ml_metrics", "paths_gate_out", "paths_gate_log"
    ]

    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for profile in profiles:
            for i in range(1, args.runs + 1):
                run_name = f"run_{i:02d}"
                snap_dir = outdir / profile / run_name
                snap_dir.mkdir(parents=True, exist_ok=True)

                # Clean working dirs each trial
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
                # allow_fail=True because HOLD/REJECT may exit non-zero intentionally
                gate_proc = run(["python", "ml/src/decision_gate.py"], env=env, allow_fail=True, capture=True)

                # Save gate logs for proof/debug
                gate_log_path = snap_dir / "gate_stdout_stderr.log"
                gate_log_path.write_text(
                    (gate_proc.stdout or "") + ("\n\n--- STDERR ---\n\n") + (gate_proc.stderr or ""),
                    encoding="utf-8"
                )

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

                # ML fields
                ml_ar = float(safe_get(ml_metrics, "anomaly_rate", 0.0) or 0.0)
                ml_f1 = float(safe_get(ml_metrics, "f1", 0.0) or 0.0)

                # Oracle + baselines
                oracle = oracle_decision(profile)
                trivy_only = trivy_only_strict_decision(critical, high, medium, low)
                ml_only = ml_only_decision(ml_ar, ml_f1, args.ml_only_ar_thresh, args.ml_only_f1_thresh)
                hybrid = str(safe_get(gate_out, "decision", "") or "")

                row = {
                    "profile": profile,
                    "run": run_name,

                    "oracle_decision": oracle,
                    "trivy_only_decision": trivy_only,
                    "ml_only_decision": ml_only,
                    "hybrid_decision": hybrid,

                    "oracle_match_trivy_only": int(trivy_only == oracle),
                    "oracle_match_ml_only": int(ml_only == oracle),
                    "oracle_match_hybrid": int(hybrid == oracle),

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
                    "high_severity_total": high_sev_total,

                    "ml_anomaly_rate": ml_ar,
                    "ml_precision": safe_get(ml_metrics, "precision", safe_get(ml_metrics, "prec", None)),
                    "ml_recall": safe_get(ml_metrics, "recall", safe_get(ml_metrics, "rec", None)),
                    "ml_f1": safe_get(ml_metrics, "f1", None),
                    "ml_auc": safe_get(ml_metrics, "roc_auc", safe_get(ml_metrics, "auc", None)),
                    "ml_pr_auc": safe_get(ml_metrics, "avg_precision", safe_get(ml_metrics, "pr_auc", None)),

                    "gate_decision": safe_get(gate_out, "decision", ""),
                    "gate_reason": safe_get(gate_out, "reason", safe_get(gate_out, "explanation", "")),
                    "gate_exit_code": gate_proc.returncode,

                    "paths_trivy_metrics": str(trivy_metrics_path),
                    "paths_ml_metrics": str(ml_metrics_path),
                    "paths_gate_out": str(gate_out_path),
                    "paths_gate_log": str(gate_log_path),
                }

                w.writerow(row)
                print(f"✅ {profile} {run_name} -> oracle={oracle} | trivy_only={trivy_only} | ml_only={ml_only} | hybrid={hybrid} (exit={gate_proc.returncode})")

    print(f"\n✅ Wrote: {detailed_csv}")


if __name__ == "__main__":
    main()
