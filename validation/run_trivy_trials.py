# validation/run_trivy_trials.py
"""
Run repeated Trivy secret-scan trials across injection profiles and store outputs under:

  validation/trivy_runs/<profile>/run_XX/

This script uses your existing generator:
  python trivy/make_payment_set_trivy.py --id <ID> --profile <profile>

and scorer:
  python trivy/score_trivy.py --scan ... --gt-csv ... --out ...

Profiles default: low, medium, high
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def require_trivy() -> None:
    if shutil.which("trivy") is None:
        raise SystemExit(
            "ERROR: trivy is not installed or not in PATH.\n"
            "If you're running via GitHub Actions, Trivy will be installed there."
        )


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def count_gt_rows(gt_csv: Path) -> int:
    if not gt_csv.exists():
        return 0
    lines = gt_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) <= 1:
        return 0
    return max(0, len(lines) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="Runs per profile (default: 10)")
    ap.add_argument(
        "--profiles",
        default="low,medium,high",
        help="Comma-separated profiles (default: low,medium,high)",
    )
    ap.add_argument(
        "--template",
        default="trivy/payment_set_template",
        help="Template folder (default: trivy/payment_set_template)",
    )
    ap.add_argument(
        "--out-root",
        default="validation/trivy_runs",
        help="Output root (default: validation/trivy_runs)",
    )
    ap.add_argument(
        "--keep-datasets",
        action="store_true",
        help="Keep datasets/payment_set_* folders (default: off; deletes them)",
    )
    args = ap.parse_args()

    # If running locally, require Trivy. In GitHub Actions, Trivy is installed in the workflow.
    require_trivy()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    profiles = [p.strip().lower() for p in args.profiles.split(",") if p.strip()]
    if not profiles:
        raise SystemExit("ERROR: No profiles provided.")

    make_payment = repo_root / "trivy" / "make_payment_set_trivy.py"
    score_trivy = repo_root / "trivy" / "score_trivy.py"
    template_path = (repo_root / args.template).resolve()

    if not make_payment.exists():
        raise SystemExit(f"ERROR: Missing {make_payment}")
    if not score_trivy.exists():
        raise SystemExit(f"ERROR: Missing {score_trivy}")
    if not template_path.exists():
        raise SystemExit(f"ERROR: Missing template folder: {template_path}")

    print(f"\nRepo root: {repo_root}")
    print(f"Output   : {out_root}")
    print(f"Profiles : {profiles}")
    print(f"Runs     : {args.runs}\n")

    for profile in profiles:
        for i in range(1, args.runs + 1):
            run_name = f"run_{i:02d}"
            set_id = f"sv_{profile}_{i:02d}"

            # 1) Generate dataset (includes injection + ground truth)
            run_cmd(
                [
                    sys.executable,
                    str(make_payment),
                    "--id",
                    set_id,
                    "--profile",
                    profile,
                    "--template",
                    str(template_path),
                ],
                cwd=repo_root,
            )

            dataset_dir = repo_root / "datasets" / f"payment_set_{set_id}"
            gt_csv = dataset_dir / "ground_truth" / "secrets.csv"

            # 2) Prepare output folder for this run
            run_out = out_root / profile / run_name
            run_out.mkdir(parents=True, exist_ok=True)

            scan_json = run_out / "scan.json"
            metrics_json = run_out / "trivy_metrics.json"
            meta_json = run_out / "run_meta.json"

            # 3) Trivy scan (secrets only)
            run_cmd(
                [
                    "trivy",
                    "fs",
                    "--scanners",
                    "secret",
                    "--format",
                    "json",
                    "--exit-code",
                    "0",
                    "--no-progress",
                    "-o",
                    str(scan_json),
                    str(dataset_dir),
                ],
                cwd=repo_root,
            )

            # 4) Score vs ground truth
            run_cmd(
                [
                    sys.executable,
                    str(score_trivy),
                    "--scan",
                    str(scan_json),
                    "--gt-csv",
                    str(gt_csv),
                    "--out",
                    str(metrics_json),
                ],
                cwd=repo_root,
            )

            # 5) Metadata (for traceability)
            meta = {
                "profile": profile,
                "run": i,
                "set_id": set_id,
                "dataset_dir": str(dataset_dir),
                "ground_truth_csv": str(gt_csv),
                "ground_truth_count": count_gt_rows(gt_csv),
                "ts_unix": int(time.time()),
            }
            meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"✓ Saved: {metrics_json}")

            # 6) Cleanup dataset (recommended to avoid bloat)
            if not args.keep_datasets and dataset_dir.exists():
                shutil.rmtree(dataset_dir, ignore_errors=True)

    print("\nDONE: Trivy trials completed.")
    print(f"Outputs are in: {out_root}")


if __name__ == "__main__":
    main()
