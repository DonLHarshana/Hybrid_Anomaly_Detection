# trivy/score_trivy.py
"""
Read a Trivy JSON scan and emit compact metrics JSON for the Decision Gate.

Usage:
  python trivy/score_trivy.py --scan trivy_out/scan.json --out trivy_out/trivy_metrics.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

def collect_counts(obj, counts: Counter):
    if isinstance(obj, dict):
        vulns = obj.get("Vulnerabilities")
        if isinstance(vulns, list):
            for v in vulns:
                sev = (v.get("Severity") or "UNKNOWN").upper()
                counts[sev] += 1
        for v in obj.values():
            collect_counts(v, counts)
    elif isinstance(obj, list):
        for v in obj:
            collect_counts(v, counts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="Path to Trivy JSON scan")
    ap.add_argument("--out",  required=True, help="Path to write metrics JSON")
    args = ap.parse_args()

    scan = json.loads(Path(args.scan).read_text())
    counts = Counter()
    collect_counts(scan, counts)

    high = counts.get("HIGH", 0) + counts.get("CRITICAL", 0)
    medium = counts.get("MEDIUM", 0)
    low = counts.get("LOW", 0)

    if high > 0:
        risk = "high"
    elif medium > 0:
        risk = "medium"
    elif low > 0:
        risk = "low"
    else:
        risk = "low"

    metrics = {
        "risk": risk,
        "critical": counts.get("CRITICAL", 0),
        "high": counts.get("HIGH", 0),
        "medium": counts.get("MEDIUM", 0),
        "low": counts.get("LOW", 0),
        "unknown": counts.get("UNKNOWN", 0),
        "precision": None,
        "recall": None,
        "f1": None,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
