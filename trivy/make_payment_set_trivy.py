# trivy/make_payment_set_trivy.py
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="payment set id, e.g., 0001")
    ap.add_argument(
        "--template",
        default=None,
        help="template dir; defaults to trivy/payment_set_template"
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent   # .../trivy
    repo = here.parent

    # Canonical injector path (no legacy fallback)
    injector = here / "secrets_injector.py"
    if not injector.exists():
        raise FileNotFoundError(f"Missing injector: {injector}")

    # Canonical template dir (no legacy fallback)
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
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Generated: {out_dir}")

if __name__ == "__main__":
    main()
