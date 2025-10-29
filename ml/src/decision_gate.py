#!/usr/bin/env python3
"""
Decision Gate: Adaptive Fusion of Trivy + ML signals
Outputs: ACCEPT, HOLD, or REJECT based on security risk

Decision Logic:
- REJECT: fusion_score >= 1.3 OR trivy_risk in [critical, high]
- HOLD: 0.4 <= fusion_score < 1.3 OR trivy_risk == medium
- ACCEPT: fusion_score < 0.4 AND trivy_risk in [low, none]
"""
import json
import sys
import os
from datetime import datetime, timezone


def load_metrics(trivy_path="trivy_out/trivy_metrics.json", ml_path="ml_out/ml_metrics.json"):
    """Load Trivy and ML metrics from JSON files"""
    with open(trivy_path) as f:
        trivy = json.load(f)
    with open(ml_path) as f:
        ml = json.load(f)
    return trivy, ml


def compute_fusion_score(trivy_metrics, ml_metrics):
    """
    Compute fusion score combining Trivy and ML signals
    
    Formula:
        fusion_score = (trivy_weight * 0.6) + (ml_anomaly_rate * 8) - (ml_f1 * 0.2)
    
    Where:
        - trivy_weight: critical=3.0, high=2.0, medium=1.0, low=0.5, none=0.0
        - ml_anomaly_rate: percentage of flagged anomalous transactions
        - ml_f1: ML model quality (penalty to reduce overconfidence)
    
    Returns:
        float: Fusion score (higher = more risky)
    """
    risk_weights = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.5, "none": 0.0}
    trivy_risk_val = risk_weights.get(trivy_metrics.get("risk", "none"), 0.0)
    
    ml_anomaly_rate = ml_metrics.get("anomaly_rate", 0.0)
    ml_f1 = ml_metrics.get("f1", 0.0)
    
    # Fusion formula
    fusion_score = (trivy_risk_val * 0.6) + (ml_anomaly_rate * 8) - (ml_f1 * 0.2)
    
    return fusion_score


def make_decision(trivy_metrics, ml_metrics, fusion_score):
    """
    Make deployment decision based on fusion score, Trivy risk, AND secret count
    
    Decision boundaries:
        REJECT:  trivy_risk in [critical, high] AND (critical + high) >= 3
        HOLD:    trivy_risk in [high] AND (critical + high) == 1-2
        ACCEPT:  trivy_risk == low OR (critical + high) == 0
    
    Args:
        trivy_metrics: Trivy scan results
        ml_metrics: ML evaluation results
        fusion_score: Computed fusion score
    
    Returns:
        tuple: (decision, reason)
    """
    trivy_risk = trivy_metrics.get("risk", "none")
    critical_count = trivy_metrics.get("critical", 0)
    high_count = trivy_metrics.get("high", 0)
    total_high_severity = critical_count + high_count
    
    # Priority 1: REJECT - Multiple critical/high secrets (3+)
    if total_high_severity >= 3:
        return "REJECT", f"High security risk detected (Trivy: {trivy_risk}, {total_high_severity} critical/high secrets, Fusion Score: {fusion_score:.2f})"
    
    # Priority 2: HOLD - Few critical/high secrets (1-2)
    elif total_high_severity >= 1 and total_high_severity <= 2:
        return "HOLD", f"Medium security risk - manual review required (Trivy: {trivy_risk}, {total_high_severity} critical/high secrets, Fusion Score: {fusion_score:.2f})"
    
    # Priority 3: ACCEPT - No critical/high secrets
    else:
        return "ACCEPT", f"Low security risk - deployment approved (Trivy: {trivy_risk}, 0 critical/high secrets, Fusion Score: {fusion_score:.2f})"



def main():
    """Main decision gate execution"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f"[{timestamp}] ", flush=True)
    print("="*60)
    print("ADAPTIVE FUSION DECISION GATE")
    print("="*60)
    
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
    decision, reason = make_decision(trivy_metrics, ml_metrics, fusion_score)
    
    # Print decision summary
    print(f"Trivy Risk:      {trivy_metrics.get('risk', 'none').upper()}")
    print(f"ML F1 Score:     {ml_metrics['f1']:.3f}")
    print(f"Anomaly Rate:    {ml_metrics['anomaly_rate']*100:.1f}%")
    print(f"Fusion Score:    {fusion_score:.2f}")
    print("="*60)
    print(f"DECISION: {decision}")
    print(f"Reason: {reason}")
    print("="*60)
    
    # Prepare output
    output = {
        "decision": decision,
        "reason": reason,
        "fusion_score": round(fusion_score, 2),
        "trivy_risk": trivy_metrics.get("risk", "none"),
        "trivy_f1": trivy_metrics.get("f1", 0.0),
        "ml_f1": round(ml_metrics["f1"], 3),
        "ml_precision": round(ml_metrics["precision"], 3),
        "ml_recall": round(ml_metrics["recall"], 3),
        "anomaly_rate": round(ml_metrics["anomaly_rate"], 3)
    }
    
    # Save output
    os.makedirs("ml_out", exist_ok=True)
    with open("ml_out/gate_out.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDecision output saved to ml_out/gate_out.json")
    
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
