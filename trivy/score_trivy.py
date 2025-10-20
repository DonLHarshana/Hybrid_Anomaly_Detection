# trivy/score_trivy.py
"""
Read a Trivy JSON scan and emit compact metrics JSON for the Decision Gate.

Usage:
  # minimal (no ground truth; GT-derived fields will be null)
  python trivy/score_trivy.py --scan trivy_out/scan.json --out trivy_out/trivy_metrics.json

  # with ground truth (populates TP/FP/FN, precision/recall/f1, risk score, etc.)
  python trivy/score_trivy.py --scan trivy_out/payment_set_0001.json \
    --out trivy_out/payment_set_0001.metrics.json \
    --payment-set-id payment_set_0001 \
    --gt-high 7 --gt-medium 0 --gt-low 1 \
    --weights "0.7,0.2,0.1" 
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, Any

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

def parse_weights(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in (s or "").split(",")]
    if len(parts) != 3:
        raise ValueError("Weights must be 'high,medium,low' e.g. '0.7,0.2,0.1'")
    h, m, l = (float(parts[0]), float(parts[1]), float(parts[2]))
    return h, m, l

def safe_div(n: float, d: float):
    return (float(n) / float(d)) if d not in (0, 0.0) else None

def compute_prf(tp: int, fp: int, fn: int):
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision and recall and (precision + recall) > 0) else None
    return precision, recall, f1

def per_severity_confusion(pred_high: int, pred_med: int, pred_low: int,
                           gt_high: int, gt_med: int, gt_low: int):
    def row(p, g):
        tp = min(p, g)
        fp = max(p - g, 0)
        fn = max(g - p, 0)
        return {"tp": tp, "fp": fp, "fn": fn, "pred": p, "gt": g}

    high_row = row(pred_high, gt_high)
    med_row  = row(pred_med,  gt_med)
    low_row  = row(pred_low,  gt_low)

    TP = high_row["tp"] + med_row["tp"] + low_row["tp"]
    FP = high_row["fp"] + med_row["fp"] + low_row["fp"]
    FN = high_row["fn"] + med_row["fn"] + low_row["fn"]

    return TP, FP, FN, {
        "high": high_row,
        "medium": med_row,
        "low": low_row,
    }

def infer_payment_set_id(explicit_id: str, out_path: Path, scan_path: Path) -> str:
    if explicit_id:
        return explicit_id
    if out_path and out_path.stem:
        return out_path.stem.replace(".metrics", "")
    if scan_path and scan_path.stem:
        return scan_path.stem
    return "unknown_payment_set"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="Path to Trivy JSON scan")
    ap.add_argument("--out",  required=True, help="Path to write metrics JSON")
    # Optional extras to compute PRF etc.
    ap.add_argument("--payment-set-id", default=None, help="Identifier for the payment set (optional)")
    ap.add_argument("--gt-high", type=int, default=None, help="Ground-truth HIGH count")
    ap.add_argument("--gt-medium", type=int, default=None, help="Ground-truth MEDIUM count")
    ap.add_argument("--gt-low", type=int, default=None, help="Ground-truth LOW count")
    ap.add_argument("--n-gt", type=int, default=None, help="Override total ground truth (defaults to gt-high+gt-medium+gt-low)")
    ap.add_argument("--weights", default="0.7,0.2,0.1",
                    help="Weights for risk score as 'high,medium,low' (default '0.7,0.2,0.1')")
    args = ap.parse_args()

    scan_path = Path(args.scan)
    out_path = Path(args.out)

    scan = json.loads(scan_path.read_text())
    counts = Counter()
    collect_counts(scan, counts)

    # Severity counts
    high_only = counts.get("HIGH", 0)
    critical  = counts.get("CRITICAL", 0)
    medium    = counts.get("MEDIUM", 0)
    low       = counts.get("LOW", 0)
    unknown   = counts.get("UNKNOWN", 0)

    # Risk label
    if (high_only + critical) > 0:
        risk = "high"
    elif medium > 0:
        risk = "medium"
    elif low > 0:
        risk = "low"
    else:
        risk = "low"

    # Always parse weights so they can be included even when GT is missing
    w_high, w_med, w_low = parse_weights(args.weights)

    # ---- Defaults for GT-derived fields so keys always exist in metrics ----
    n_gt = None
    TP = FP = FN = None
    precision = recall = f1 = None
    trivy_risk_score = None

    # Compute GT-derived values if any GT is provided
    have_gt = any(v is not None for v in (args.gt_high, args.gt_medium, args.gt_low))
    if have_gt:
        gt_high = int(args.gt_high or 0)
        gt_med  = int(args.gt_medium or 0)
        gt_low  = int(args.gt_low or 0)
        n_gt = int(args.n_gt) if args.n_gt is not None else (gt_high + gt_med + gt_low)

        TP, FP, FN, _per_sev = per_severity_confusion(
            pred_high=high_only + critical,  # treat CRITICAL as HIGH for matching
            pred_med=medium,
            pred_low=low,
            gt_high=gt_high,
            gt_med=gt_med,
            gt_low=gt_low
        )

        precision, recall, f1 = compute_prf(TP, FP, FN)

        weighted_sum = (w_high * (high_only + critical)) + (w_med * medium) + (w_low * low)
        trivy_risk_score = (weighted_sum / n_gt) if n_gt and n_gt > 0 else None
        if trivy_risk_score is not None:
            trivy_risk_score = max(0.0, min(1.0, float(trivy_risk_score)))

    # ---- Build metrics dict AFTER variables are defined (or defaulted) ----
    metrics: Dict[str, Any] = {
        "payment_set_id": infer_payment_set_id(args.payment_set_id, out_path, scan_path),
        "risk": risk,
        "critical": critical,
        "high": high_only,
        "medium": medium,
        "low": low,
        "unknown": unknown,
        "n_gt": n_gt,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
        "f1": round(f1, 6) if f1 is not None else None,
        "trivy_risk_score": round(trivy_risk_score, 6) if trivy_risk_score is not None else None,
        "weights": {"high": w_high, "medium": w_med, "low": w_low},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
