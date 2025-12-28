#!/usr/bin/env python3
import argparse, csv, json, os, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd, env=None):
    r = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}")
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--profiles", default="clean,low,medium,high")
    args = ap.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    runs = int(args.runs)

    out_dir = ROOT / "validation" / "gitleaks_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = out_dir / "gitleaks_runs_detailed.csv"

    # header
    if not detailed_csv.exists():
        with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "profile","run","risk","findings","TP","FP","FN","precision","recall","f1"
            ])
            w.writeheader()

    for profile in profiles:
        for i in range(1, runs + 1):
            # clean dirs each run
            for d in ["datasets", "gitleaks_out", "trivy_out", "ml_out", "artifacts"]:
                p = ROOT / d
                if p.exists():
                    shutil.rmtree(p)
            (ROOT / "datasets").mkdir(parents=True, exist_ok=True)
            (ROOT / "gitleaks_out").mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env["INJECT_PROFILE"] = profile

            # generate payment set (reuses your existing injector + GT csv)
            run(["python", "trivy/make_payment_set_trivy.py", "--id", "0001", "--template", "trivy/payment_set_template"], env=env)

            # gitleaks scan the generated folder (directory scan)
            run([
                "gitleaks", "dir", "datasets/payment_set_0001",
                "--report-format", "json",
                "--report-path", "gitleaks_out/report.json",
                "--exit-code", "0",
                "--no-banner",
                "--redact"
            ], env=env)

            # score vs ground truth
            run([
                "python", "gitleaks/score_gitleaks.py",
                "--report", "gitleaks_out/report.json",
                "--gt-csv", "datasets/payment_set_0001/ground_truth/secrets.csv",
                "--out", "gitleaks_out/gitleaks_metrics.json"
            ], env=env)

            metrics = json.loads((ROOT / "gitleaks_out" / "gitleaks_metrics.json").read_text(encoding="utf-8"))

            row = {
                "profile": profile,
                "run": i,
                "risk": metrics.get("risk"),
                "findings": metrics.get("findings"),
                "TP": metrics.get("TP"),
                "FP": metrics.get("FP"),
                "FN": metrics.get("FN"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
            }

            with open(detailed_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=row.keys())
                w.writerow(row)

            print(f"✅ {profile} run_{i:02d} -> findings={row['findings']} f1={row['f1']}")

if __name__ == "__main__":
    main()
