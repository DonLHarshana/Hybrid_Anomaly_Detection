#!/usr/bin/env python3
# trivy/score_trivy.py
"""
Read a Trivy JSON scan and emit compact metrics JSON for the Decision Gate.

This project validates *secret scanning*, so:
- Severity counts are taken from Secrets[] only.
- Precision/Recall/F1 are computed by comparing detected secrets vs ground-truth secrets.csv.
- risk is derived from the same critical/high/medium/low counts (so it cannot disagree).

Risk mapping aligned with Decision Gate:
- (critical+high) == 0  -> low
- (critical+high) 1..2  -> medium
- (critical+high) >= 3  -> high
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Tuple, Set, Optional

SEVS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]


def safe_div(n: float, d: float) -> float:
    return (float(n) / float(d)) if d not in (0, 0.0) else 0.0


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


def count_secret_severities(scan: dict) -> Counter:
    """
    Count severities ONLY from secret findings:
      scan["Results"][i]["Secrets"][j]["Severity"]
    """
    counts = Counter()
    for r in scan.get("Results", []):
        for f in (r.get("Secrets") or []):
            sev = (f.get("Severity") or "UNKNOWN").upper()
            counts[sev] += 1
    for s in SEVS:
        counts.setdefault(s, 0)
    return counts


def risk_bucket_from_counts(critical: int, high: int, medium: int, low: int) -> str:
    """
    Match Decision Gate behaviour:
      0 -> low
      1-2 -> medium
      3+ -> high
    """
    major = int(critical) + int(high)
    if major >= 3:
        return "high"
    if major >= 1:
        return "medium"
    # If no major but there are medium/low findings, still call it low
    if int(medium) > 0 or int(low) > 0:
        return "low"
    return "low"


def load_gt_csv(gt_csv: Path) -> Set[Tuple[str, str]]:
    rows: Set[Tuple[str, str]] = set()
    if not gt_csv.exists():
        return rows
    with open(gt_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fp = (row.get("file_path") or "").strip()
            st = _norm_secret_type(row.get("secret_type", ""))
            if fp:
                rows.add((fp, st))
    return rows


def _to_rel_target(target: str, dataset_root: Optional[Path]) -> str:
    """
    Normalize Trivy 'Target' (often full path) to a relative path like in ground truth (Dockerfile, configs/..).
    """
    t = (target or "").strip()
    if not t:
        return ""

    p = Path(t)

    # If we know dataset root (payment_set folder), strip it
    if dataset_root is not None:
        try:
            return str(p.relative_to(dataset_root))
        except Exception:
            pass

        # Sometimes Target includes ".../payment_set_0001/..." but not as Path relative_to (string mismatch).
        marker = str(dataset_root).replace("\\", "/") + "/"
        t2 = t.replace("\\", "/")
        if marker in t2:
            return t2.split(marker, 1)[1]

    # fallback: keep only last 2 parts to preserve subdirs like configs/config.yml
    parts = p.parts
    if len(parts) >= 2:
        return str(Path(parts[-2]) / parts[-1])
    return p.name or t


def found_secret_keyset(scan: dict, dataset_root: Optional[Path]) -> Set[Tuple[str, str]]:
    """
    Build set of detected secrets as (relative_file_path, normalized_secret_type).
    """
    keys: Set[Tuple[str, str]] = set()
    for r in scan.get("Results", []):
        target_raw = r.get("Target", "")
        target = _to_rel_target(target_raw, dataset_root)

        for f in (r.get("Secrets") or []):
            rule = f.get("RuleID") or f.get("Title") or ""
            keys.add((target, _norm_secret_type(rule)))
    return keys


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--scan", required=True, help="Path to trivy_out/scan.json")
    ap.add_argument("--out", required=True, help="Path to write trivy_out/trivy_metrics.json")
    ap.add_argument("--gt-csv", help="Path to datasets/.../ground_truth/secrets.csv")
    args = ap.parse_args()

    scan_path = Path(args.scan)
    out_path = Path(args.out)

    scan = json.loads(scan_path.read_text(encoding="utf-8"))

    # Derive dataset root from gt-csv location: .../payment_set_0001/ground_truth/secrets.csv
    dataset_root = None
    if args.gt_csv:
        gt_path = Path(args.gt_csv)
        if gt_path.exists():
            dataset_root = gt_path.parent.parent  # payment_set_XXXX

    # 1) Count severities (Secrets only)
    counts = count_secret_severities(scan)

    critical = int(counts["CRITICAL"])
    high = int(counts["HIGH"])
    medium = int(counts["MEDIUM"])
    low = int(counts["LOW"])
    unknown = int(counts["UNKNOWN"])

    risk = risk_bucket_from_counts(critical, high, medium, low)

    metrics: Dict[str, Any] = {
        "risk": risk,
        "critical": critical,
        "high": high,
        "medium": medium,
        "low": low,
        "unknown": unknown,

        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,  # not meaningful for secret scanning; keep 0
    }

    # 2) Precision/Recall/F1 from ground truth (oracle)
    if args.gt_csv:
        gt_path = Path(args.gt_csv)
        if gt_path.exists():
            gt = load_gt_csv(gt_path)
            found = found_secret_keyset(scan, dataset_root)

            TP = len(found & gt)
            FP = len(found - gt)
            FN = len(gt - found)
            TN = 0

            prec = safe_div(TP, TP + FP)
            rec = safe_div(TP, TP + FN)
            f1 = safe_div(2 * prec * rec, (prec + rec))

            metrics.update({
                "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
            })

    # Safety guard: if critical/high exist, risk must be at least medium/high
    major = metrics["critical"] + metrics["high"]
    if major >= 3:
        metrics["risk"] = "high"
    elif major >= 1:
        metrics["risk"] = "medium"
    else:
        metrics["risk"] = "low"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
