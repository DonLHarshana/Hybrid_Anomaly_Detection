#!/usr/bin/env python3
import argparse, json, csv, os
from collections import Counter
from pathlib import Path

SEVERITIES = ["CRITICAL","HIGH","MEDIUM","LOW","UNKNOWN"]

def norm(p): return Path(p).as_posix()

def collect_counts_and_secrets(scan_path):
    counts = Counter({s:0 for s in SEVERITIES})
    secrets = []
    data = json.loads(Path(scan_path).read_text(encoding="utf-8"))
    for res in (data.get("Results") or []):
        for v in (res.get("Vulnerabilities") or []): counts[(v.get("Severity") or "UNKNOWN").upper()] += 1
        for m in (res.get("Misconfigurations") or []): counts[(m.get("Severity") or "UNKNOWN").upper()] += 1
        for s in (res.get("Secrets") or []):
            counts[(s.get("Severity") or "UNKNOWN").upper()] += 1
            secrets.append((norm(s.get("File","")), int(s.get("StartLine",0)), int(s.get("EndLine",0)), (s.get("RuleID") or "").strip()))
    return counts, secrets

def load_gt(gt_csv_path):
    if not gt_csv_path or not os.path.exists(gt_csv_path): return []
    rows = []
    with open(gt_csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = [c.lower().strip() for c in (r.fieldnames or [])]
        has_range = all(k in cols for k in ["file","start_line","end_line"])
        has_line_only = all(k in cols for k in ["type","file","line"])
        for row in r:
            if has_range:
                rows.append((norm(row["file"]), int(row["start_line"]), int(row["end_line"]), (row.get("rule_id") or "").strip()))
            elif has_line_only:
                line = int(row["line"]); rows.append((norm(row["file"]), line, line, (row.get("type") or "").strip()))
    return rows

def overlap(a1,a2,b1,b2): return not (a2 < b1 or b2 < a1)

def tp_fp_fn(gt, det):
    used_gt, used_det = set(), set()
    for i,(df,ds,de,_) in enumerate(det):
        for j,(gf,gs,ge,_) in enumerate(gt):
            if j in used_gt: continue
            if df == gf and overlap(ds,de,gs,ge):
                used_gt.add(j); used_det.add(i); break
    tp = len(used_gt); fp = len(det) - len(used_det); fn = len(gt) - len(used_gt)
    return tp, fp, fn

def prf(tp, fp, fn):
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = (2*p*r/(p+r)) if (p+r)>0 else 0.0
    return p, r, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-secrets")
    args = ap.parse_args()

    counts, det_secrets = collect_counts_and_secrets(args.scan)
    total = sum(counts.values())
    risk = ("high" if (counts["CRITICAL"] or counts["HIGH"]) else
            "medium" if counts["MEDIUM"] else
            "low" if total>0 else "none")

    precision = recall = f1 = "n/a"
    if args.gt_secrets and os.path.exists(args.gt_secrets):
        gt = load_gt(args.gt_secrets)
        tp, fp, fn = tp_fp_fn(gt, det_secrets)
        precision, recall, f1 = prf(tp, fp, fn)

    metrics = {
        "risk": risk,
        "critical": counts["CRITICAL"],
        "high": counts["HIGH"],
        "medium": counts["MEDIUM"],
        "low": counts["LOW"],
        "unknown": counts["UNKNOWN"],
        "secret_hits": len(det_secrets),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
