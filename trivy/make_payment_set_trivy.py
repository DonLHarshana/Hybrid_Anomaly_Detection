# trivy/make_payment_set_trivy.py
import argparse
import subprocess
import sys
import os
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="payment set id, e.g., 0001")
    ap.add_argument(
        "--template",
        default=None,
        help="template dir; defaults to trivy/payment_set_template"
    )
    # Optional: control injection profile; forwarded to secrets_injector.py
    ap.add_argument(
        "--profile",
        choices=["low", "medium", "high"],
        default=os.getenv("INJECT_PROFILE", "medium").lower(),
        help="injection profile passed to secrets_injector.py via env (low|medium|high)"
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent   # .../trivy
    repo = here.parent

    # Canonical injector path
    injector = here / "secrets_injector.py"
    if not injector.exists():
        raise FileNotFoundError(f"Missing injector: {injector}")

    # Canonical template dir
    template = Path(args.template) if args.template else (here / "payment_set_template")
    if not template.exists():
        raise FileNotFoundError(f"Missing template dir: {template}")

    out_dir = repo / "datasets" / f"payment_set_{args.id}"
    gt_csv  = out_dir / "ground_truth" / "secrets.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(injector),
        "--template", str(template),
        "--out", str(out_dir),
        "--gt", str(gt_csv),
    ]
    env = os.environ.copy()
    env["INJECT_PROFILE"] = args.profile  # forward profile to injector

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)
    print(f"Generated: {out_dir}")

    # --- breadcrumb so the scorer can auto-find the right ground_truth ---
    run_ctx = repo / "datasets" / "run_context.json"
    try:
        run_ctx.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_payment_set_id": args.id,
            "last_dataset_path": str(out_dir),
            "ground_truth_csv": str(gt_csv),
            "profile": args.profile,
        }
        with open(run_ctx, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote {run_ctx} -> GT: {gt_csv}")
    except Exception as e:
        print(f"[WARN] could not write {run_ctx}: {e}")

if __name__ == "__main__":
    main()
