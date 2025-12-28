#!/usr/bin/env python3
"""
Score Gitleaks JSON report against ground truth secrets.csv (same format you use for Trivy scoring).

Usage:
  python gitleaks/score_gitleaks.py \
    --report gitleaks_out/report.json \
    --gt-csv datasets/payment_set_0001/ground_truth/secrets.csv \
    --out gitleaks_out/gitleaks_metrics.json
"""

import argparse, json, csv
from pathlib import Path
from typing import Dict, Any, Set, Tuple


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d not in (0, 0.0) else 0.0


def normalize_fp(p: str) -> str:
    """Make paths comparable to your GT (strip datasets/payment_set_0001 prefixes etc.)."""
    p = (p or "").replace("\\", "/").lstrip("./")
    # strip any leading workspace parts
    marker = "datasets/payment_set_0001/"
    if marker in p:
        p = p.split(marker, 1)[1]
    if p.startswith("payment_set_0001/"):
        p = p.split("payment_set_0001/", 1)[1]
    return p


def _norm_secret_type(s: str) -> str:
    s = (s or "").lower()
    if "aws" in s and ("access" in s or "key" in s or "akia" in s):
        return "aws_access_key_id"
    if "aws" in s and "secret" in s:
        return "aws_secret_access_key"
    if "postgres" in s or "postgresql" in s:
        return "postgres_uri"
    if "jwt" in s or "json web token" in s:
        return "dummy_jwt"
    if "api" in s or "generic" in s or "stripe" in s or "sk_test" in s:
        return "generic_api_key"
    return s or "unknown"


def load_gt_csv(gt_csv: Path) -> Set[Tuple[str, str]]:
    rows = set()
    if not gt_csv.exists():
        return rows
    with open(gt_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fp = normalize_fp(row.get("file_path", ""))
            st = _norm_secret_type(row.get("secret_type", ""))
            rows.add((fp, st))
    return rows


def parse_gitleaks_report(report_path: Path) -> list:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    # Some versions output a list; others may wrap in an object.
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("Leaks") or data.get("leaks") or data.get("findings") or []
    return []


def found_keyset_from_gitleaks(findings: list) -> Set[Tuple[str, str]]:
    keys = set()
    for f in findings:
        if not isinstance(f, dict):
            continue
        fp = normalize_fp(f.get("File") or f.get("file") or f.get("Path") or f.get("path") or "")
        rule = f.get("RuleID") or f.get("RuleId") or f.get("rule") or ""
        desc = f.get("Description") or f.get("description") or ""
        keys.add((fp, _norm_secret_type(rule or desc)))
    return keys


def risk_bucket(found_count: int) -> str:
    # simple bucket just for reporting
    if found_count >= 3:
        return "high"
    if found_count >= 1:
        return "medium"
    return "low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to gitleaks JSON report")
    ap.add_argument("--gt-csv", required=True, help="Path to ground_truth/secrets.csv")
    ap.add_argument("--out", required=True, help="Path to write gitleaks_metrics.json")
    args = ap.parse_args()

    report_path = Path(args.report)
    gt_path = Path(args.gt_csv)
    out_path = Path(args.out)

    findings = parse_gitleaks_report(report_path) if report_path.exists() else []
    gt = load_gt_csv(gt_path)
    found = found_keyset_from_gitleaks(findings)

    TP = len(found & gt)
    FP = len(found - gt)
    FN = len(gt - found)
    TN = 0

    prec = safe_div(TP, TP + FP)
    rec = safe_div(TP, TP + FN)
    f1 = safe_div(2 * prec * rec, (prec + rec) if (prec + rec) else 0.0)

    metrics: Dict[str, Any] = {
        "tool": "gitleaks",
        "risk": risk_bucket(len(found)),
        "findings": len(found),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
