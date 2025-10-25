#!/usr/bin/env python3
# trivy/score_trivy.py
"""
Read a Trivy JSON scan and emit compact metrics JSON for the Decision Gate.

Usage (with CSV ground truth):
  python trivy/score_trivy.py \
    --scan trivy_out/scan.json \
    --gt-csv datasets/payment_set_0001/ground_truth/secrets.csv \
    --out trivy_out/trivy_metrics.json

Back-compat (numeric GT counts):
  python trivy/score_trivy.py --scan ... --out ... \
    --payment-set-id payment_set_0001 \
    --gt-high 7 --gt-medium 0 --gt-low 1 \
    --weights "0.7,0.2,0.1"
"""
import argparse, json, csv
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Tuple, Set


SEVS = ["CRITICAL","HIGH","MEDIUM","LOW","UNKNOWN"]


def parse_weights(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in (s or "").split(",")]
    if len(parts) != 3:
        raise ValueError("Weights must be 'high,medium,low' e.g. '0.7,0.2,0.1'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def count_severities(scan: dict) -> Counter:
    counts = Counter()
    for r in scan.get("Results", []):
        for coll in ("Vulnerabilities", "Misconfigurations", "Secrets", "Licenses"):
            for f in (r.get(coll) or []):
                sev = (f.get("Severity") or "UNKNOWN").upper()
                counts[sev] += 1
    # ensure keys exist
    for s in SEVS:
        counts.setdefault(s, 0)
    return counts


def risk_bucket(c: Counter) -> str:
    critical = c.get("CRITICAL", 0)
    high = c.get("HIGH", 0)
    medium = c.get("MEDIUM", 0)
    low = c.get("LOW", 0)

    # Logic: Highest severity present determines risk
    if critical > 0:
        return "high"
    elif high > 0:
        return "high"
    elif medium > 0:
        return "medium"
    elif low > 0:
        return "low"
    else:
        # No findings at all
        return "low"



# ---------- Secrets P/R/F1 helpers ----------
def _norm_secret_type(s: str) -> str:
    s = (s or "").lower()
    if "aws" in s and ("access" in s or "key" in s):
        return "aws_access_key_id"
    if "aws" in s and "secret" in s:
        return "aws_secret_access_key"
    if "postgres" in s or "postgresql" in s:
        return "postgres_uri"
    if "jwt" in s or "json web token" in s:
        return "dummy_jwt"
    if "api" in s or "generic" in s or "stripe" in s:
        return "generic_api_key"
    return s or "unknown"


def found_secret_keyset(scan: dict) -> Set[Tuple[str, str]]:
    keys = set()
    for r in scan.get("Results", []):
        target = r.get("Target", "")
        for f in (r.get("Secrets") or []):
            rule = f.get("RuleID") or f.get("Title") or ""
            keys.add((target, _norm_secret_type(rule)))
    return keys


def load_gt_csv(gt_csv: Path) -> Set[Tuple[str, str]]:
    rows = set()
    if not gt_csv.exists(): return rows
    with open(gt_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            fp = row.get("file_path", "")
            st = _norm_secret_type(row.get("secret_type", ""))
            rows.add((fp, st))
    return rows


def safe_div(n: float, d: float):
    return (float(n) / float(d)) if d not in (0, 0.0) else 0.0


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--scan", required=True, help="Path to trivy_out/scan.json")
    ap.add_argument("--out",  required=True, help="Path to write trivy_out/trivy_metrics.json")
    ap.add_argument("--payment-set-id", default=None)

    # NEW: CSV ground truth (preferred)
    ap.add_argument("--gt-csv", help="Path to datasets/.../ground_truth/secrets.csv")

    # Back-compat numeric GT options (do not compute P/R/F1 by type)
    ap.add_argument("--gt-high", type=int, default=None)
    ap.add_argument("--gt-medium", type=int, default=None)
    ap.add_argument("--gt-low", type=int, default=None)
    ap.add_argument("--n-gt", type=int, default=None)

    ap.add_argument("--weights", default="0.7,0.2,0.1", help="Weights 'high,medium,low'")
    args = ap.parse_args()

    scan_path = Path(args.scan)
    out_path  = Path(args.out)
    scan = json.loads(scan_path.read_text())

    # Severity counts + risk
    counts = count_severities(scan)
    risk = risk_bucket(counts)

    # Base metrics (SIMPLIFIED - only essential fields)
    metrics: Dict[str, Any] = {
        "risk": risk,
        "critical": counts["CRITICAL"],
        "precision": None,
        "recall": None,
        "f1": None,
        "TP": None,
        "FP": None,
        "FN": None,
        "TN": 0  # Trivy doesn't naturally compute TN, default to 0
    }

    # ----- Preferred path: CSV ground truth → compute P/R/F1 for Secrets -----
    if args.gt_csv:
        gt_path = Path(args.gt_csv)
        if gt_path.exists():
            gt = load_gt_csv(gt_path)
            found = found_secret_keyset(scan)
            TP = len(found & gt)
            FP = len(found - gt)
            FN = len(gt - found)
            TN = 0  # True Negatives not applicable for secret detection
            prec = safe_div(TP, TP + FP)
            rec  = safe_div(TP, TP + FN)
            f1   = safe_div(2 * prec * rec, (prec + rec) if (prec + rec) else 0.0)
            metrics.update({
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
            })

    # ----- Back-compat numeric GT (if CSV not provided) -----
    elif any(v is not None for v in (args.gt_high, args.gt_medium, args.gt_low)):
        gt_high = int(args.gt_high or 0)
        gt_med  = int(args.gt_medium or 0)
        gt_low  = int(args.gt_low or 0)
        # Treat CRITICAL as HIGH
        pred_high = counts["HIGH"] + counts["CRITICAL"]
        pred_med  = counts["MEDIUM"]
        pred_low  = counts["LOW"]
        # simple overlap by count (no per-file matching possible)
        TP = min(pred_high, gt_high) + min(pred_med, gt_med) + min(pred_low, gt_low)
        FP = max(pred_high - gt_high, 0) + max(pred_med - gt_med, 0) + max(pred_low - gt_low, 0)
        FN = max(gt_high - pred_high, 0) + max(gt_med - pred_med, 0) + max(gt_low - pred_low, 0)
        TN = 0  # Not applicable
        prec = safe_div(TP, TP + FP)
        rec  = safe_div(TP, TP + FN)
        f1   = safe_div(2 * prec * rec, (prec + rec) if (prec + rec) else 0.0)
        metrics.update({
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
