#!/usr/bin/env python3
"""
Decision Gate: Adaptive Fusion of Trivy + ML signals
Outputs: ACCEPT, HOLD, or REJECT based on security risk + ML anomalies

Decision Logic (v2 - WITH ML OVERRIDE):
- REJECT: critical+high secrets >= 3
- HOLD:   critical+high secrets = 1-2
- ACCEPT: secrets = 0 AND low ML anomaly signal

Important fix:
- Do NOT trust trivy_metrics["risk"] (it can be inconsistent if scoring logic changes).
- Always derive risk from counts (critical/high/medium/low) for stable decisions and logs.
"""

import json
import sys
import os
from datetime import datetime, timezone


def load_metrics(trivy_path="trivy_out/trivy_metrics.json", ml_path="ml_out/ml_metrics.json"):
    """Load Trivy and ML metrics from JSON files"""
    with open(trivy_path, encoding="utf-8") as f:
        trivy = json.load(f)
    with open(ml_path, encoding="utf-8") as f:
        ml = json.load(f)
    return trivy, ml


def derive_trivy_risk_from_counts(trivy_metrics) -> str:
    """
    Derive risk purely from severity counts.
    Aligned with Decision Gate thresholds:
      0 major (critical+high) -> low
      1-2 major             -> medium
      3+ major              -> high
    """
    critical = int(trivy_metrics.get("critical", 0) or 0)
    high = int(trivy_metrics.get("high", 0) or 0)
    medium = int(trivy_metrics.get("medium", 0) or 0)
    low = int(trivy_metrics.get("low", 0) or 0)

    major = critical + high
    if major >= 3:
        return "high"
    if major >= 1:
        return "medium"
    # if only medium/low findings exist, we keep it low (this project focuses secrets)
    if medium > 0 or low > 0:
        return "low"
    return "low"


def compute_fusion_score(trivy_metrics, ml_metrics):
    """
    Compute fusion score combining Trivy and ML signals

    Formula:
        fusion_score = (trivy_weight * 0.6) + (ml_anomaly_rate * 8) - (ml_f1 * 0.2)

    Where:
        - trivy_weight derived from derived_trivy_risk_from_counts(): high/medium/low
        - ml_anomaly_rate: percentage of flagged anomalous transactions
        - ml_f1: ML model quality (penalty to reduce overconfidence)

    Returns:
        float: Fusion score (higher = more risky)
    """
    # weights for risk LABELS (not severities)
    risk_weights = {"high": 3.0, "medium": 1.5, "low": 0.5, "none": 0.0}

    derived_risk = derive_trivy_risk_from_counts(trivy_metrics)
    trivy_risk_val = risk_weights.get(derived_risk, 0.0)

    ml_anomaly_rate = float(ml_metrics.get("anomaly_rate", 0.0) or 0.0)
    ml_f1 = float(ml_metrics.get("f1", 0.0) or 0.0)

    fusion_score = (trivy_risk_val * 0.6) + (ml_anomaly_rate * 8) - (ml_f1 * 0.2)
    return fusion_score


def make_decision(trivy_metrics, ml_metrics, fusion_score):
    """
    Make deployment decision based on secret count + ML anomaly override

    Decision hierarchy (v2):
        Priority 1: Secret count (Trivy)
            REJECT:  (critical + high) >= 3
            HOLD:    (critical + high) = 1-2

        Priority 2: ML anomaly override
            HOLD:    secrets = 0 AND (anomaly_rate > 10% OR (f1 > 0.30 AND recall > 0.25))

        Priority 3: ACCEPT if all clear
            ACCEPT:  secrets = 0 AND low ML signal

    Returns:
        tuple: (decision, reason, derived_trivy_risk)
    """
    critical_count = int(trivy_metrics.get("critical", 0) or 0)
    high_count = int(trivy_metrics.get("high", 0) or 0)
    total_high_severity = critical_count + high_count

    derived_trivy_risk = derive_trivy_risk_from_counts(trivy_metrics)

    # Extract ML metrics safely
    ml_anomaly_rate = float(ml_metrics.get("anomaly_rate", 0.0) or 0.0)
    ml_f1 = float(ml_metrics.get("f1", 0.0) or 0.0)
    ml_recall = float(ml_metrics.get("recall", 0.0) or 0.0)

    # Priority 1: REJECT
    if total_high_severity >= 3:
        return (
            "REJECT",
            f"High security risk detected (Trivy: {derived_trivy_risk}, {total_high_severity} critical/high secrets, Fusion Score: {fusion_score:.2f})",
            derived_trivy_risk
        )

    # Priority 2: HOLD
    if 1 <= total_high_severity <= 2:
        return (
            "HOLD",
            f"Medium security risk - manual review required (Trivy: {derived_trivy_risk}, {total_high_severity} critical/high secrets, Fusion Score: {fusion_score:.2f})",
            derived_trivy_risk
        )

    # Priority 3: ML override
    if total_high_severity == 0 and (ml_anomaly_rate > 0.10 or (ml_f1 > 0.30 and ml_recall > 0.25)):
        return (
            "HOLD",
            f"High fraud signal detected - manual review required (ML anomaly rate: {ml_anomaly_rate:.1%}, F1: {ml_f1:.3f}, Recall: {ml_recall:.3f}, Fusion Score: {fusion_score:.2f})",
            derived_trivy_risk
        )

    # Priority 4: ACCEPT
    return (
        "ACCEPT",
        f"Low security risk - deployment approved (Trivy: {derived_trivy_risk}, 0 critical/high secrets, ML anomaly rate: {ml_anomaly_rate:.1%}, Fusion Score: {fusion_score:.2f})",
        derived_trivy_risk
    )


def main():
    """Main decision gate execution"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f"[{timestamp}] Decision Gate v2 - WITH ML Override", flush=True)
    print("=" * 60)
    print("ADAPTIVE FUSION DECISION GATE")
    print("=" * 60)

    # Load metrics
    try:
        trivy_metrics, ml_metrics = load_metrics()
    except FileNotFoundError as e:
        print(f"ERROR: Missing metrics file: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in metrics file: {e}")
        sys.exit(1)

    # Compute fusion score
    fusion_score = compute_fusion_score(trivy_metrics, ml_metrics)

    # Make decision
    decision, reason, derived_trivy_risk = make_decision(trivy_metrics, ml_metrics, fusion_score)

    # Print decision summary (derived risk)
    print(f"Trivy Risk:      {derived_trivy_risk.upper()}")
    print(f"Secrets Found:   {int(trivy_metrics.get('critical', 0) or 0)} critical, {int(trivy_metrics.get('high', 0) or 0)} high")
    print(f"ML F1 Score:     {float(ml_metrics.get('f1', 0.0) or 0.0):.3f}")
    print(f"ML Recall:       {float(ml_metrics.get('recall', 0.0) or 0.0):.3f}")
    print(f"Anomaly Rate:    {float(ml_metrics.get('anomaly_rate', 0.0) or 0.0)*100:.1f}%")
    print(f"Fusion Score:    {fusion_score:.2f}")
    print("=" * 60)
    print(f"DECISION: {decision}")
    print(f"Reason: {reason}")
    print("=" * 60)

    # Prepare output
    output = {
        "decision": decision,
        "reason": reason,
        "fusion_score": round(fusion_score, 2),

        # Store derived risk so it matches counts always
        "trivy_risk": derived_trivy_risk,
        "trivy_secrets_critical": int(trivy_metrics.get("critical", 0) or 0),
        "trivy_secrets_high": int(trivy_metrics.get("high", 0) or 0),
        "trivy_f1": float(trivy_metrics.get("f1", 0.0) or 0.0),

        "ml_f1": round(float(ml_metrics.get("f1", 0.0) or 0.0), 3),
        "ml_precision": round(float(ml_metrics.get("precision", 0.0) or 0.0), 3),
        "ml_recall": round(float(ml_metrics.get("recall", 0.0) or 0.0), 3),
        "anomaly_rate": round(float(ml_metrics.get("anomaly_rate", 0.0) or 0.0), 3),

        "ml_override_triggered": (
            (int(trivy_metrics.get("critical", 0) or 0) + int(trivy_metrics.get("high", 0) or 0) == 0) and
            (float(ml_metrics.get("anomaly_rate", 0.0) or 0.0) > 0.10 or
             (float(ml_metrics.get("f1", 0.0) or 0.0) > 0.30 and float(ml_metrics.get("recall", 0.0) or 0.0) > 0.25))
        )
    }

    # Save output
    os.makedirs("ml_out", exist_ok=True)
    with open("ml_out/gate_out.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nDecision output saved to ml_out/gate_out.json")

    # Exit codes based on decision
    if decision == "REJECT":
        print("Exiting with failure code (1) — pipeline blocked", flush=True)
        sys.exit(1)
    elif decision == "HOLD":
        print("Exiting with warning code (0) — manual review required", flush=True)
        sys.exit(0)  # Don't fail build for HOLD
    else:  # ACCEPT
        print("Exiting with success code (0) — deployment approved", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
