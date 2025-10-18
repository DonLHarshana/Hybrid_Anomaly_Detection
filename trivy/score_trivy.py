#!/usr/bin/env python3
"""
Read a Trivy JSON scan and emit compact metrics JSON for the Decision Gate.

Usage:
  python trivy/score_trivy.py --scan trivy_out/scan.json --out trivy_out/trivy_metrics.json \
    [--gt-secrets datasets/payment_set_0001/ground_truth/secrets.csv]
"""
import argparse, json, csv, os
from collections import Counter
from pathlib import Path

SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]

def norm_path(p: str) -> str:
    # Normalize path + unify slashes so GT and Trivy match reliably on Windows/Linux
    return Path(p).as_posix()

def collect_counts_and_secrets(scan_json_path):
    """
    Returns:
      counts: Counter of severities across Vulnerabilities, Misconfigurations, Secrets
      secrets: set of (file, start_line, end_line, rule_id) from Trivy scan
    """
    counts = Counter({s: 0 for s in SEVERITIES})
    secrets = set()

    data = json.loads(Path(scan_json_path).read_text(encoding="utf-8"))

    for res in data.get("Results", []):
        # Vulnerabilities
        for v in (res.get("Vulnerabilities") or []):
            sev = (v.get("Severity") or "UNKNOWN").upper()
            counts[sev] += 1

        # Misconfigurations
        for m in (res.get("Misconfigurations") or []):
            sev = (m.get("Severity") or "UNKNOWN").upper()
            counts[sev] += 1

        # Secrets
        for s in (res.get("Secrets") or []):
            sev = (s.get("Severity") or "UNKNOWN").upper()
            counts[sev] += 1
            secrets.add((
                norm_path(s.get("File", "")),
                int(s.get("StartLine", 0)),
                int(s.get("EndLine", 0)),
                (s.get("RuleID") or "").strip()
            ))

    return counts, secrets

def load_gt_secrets(gt_csv_path):
    """
    CSV columns expected: file,start_line,end_line,rule_id
    """
    if not gt_csv_path or not os.path.exists(gt_csv_path):
        return set()
    gt = set()
    with open(gt_csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            gt.add((
                norm_path(row["file"]),
                int(row["start_line"]),
                int(row["end_line"]),
                (row.get("rule_id") or "").strip()
            ))
    return gt

def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="Path to Trivy JSON scan")
    ap.add_argument("--out", required=True, help="Path to write metrics JSON")
    ap.add_argument("--gt-secrets", help="CSV ground truth for secrets: file,start_line,end_line,rule_id")
    args = ap.parse_args()

    counts, trivy_secrets = collect_counts_and_secrets(args.scan)

    total = sum(counts.values())
    risk = ("high" if (counts["CRITICAL"] or counts["HIGH"]) else
            "medium" if counts["MEDIUM"] else
            "low" if total > 0 else
            "none")

    # Default to "n/a" unless GT is provided
    precision = recall = f1 = "n/a"
    if args.gt_secrets and os.path.exists(args.gt_secrets):
        gt_secrets = load_gt_secrets(args.gt_secrets)
        tp = len(gt_secrets & trivy_secrets)
        fp = len(trivy_secrets - gt_secrets)
        fn = len(gt_secrets - trivy_secrets)
        precision, recall, f1 = prf(tp, fp, fn)

    metrics = {
        "risk": risk,
        "critical": counts["CRITICAL"],
        "high": counts["HIGH"],
        "medium": counts["MEDIUM"],
        "low": counts["LOW"],
        "unknown": counts["UNKNOWN"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
