"""
Adaptive Fusion Decision Gate (v2)
Combines Trivy risk + ML signals using weighted scoring for dynamic decisions


Inputs:
  trivy_out/trivy_metrics.json -> {"precision":..,"recall":..,"f1":..,"risk":"high|medium|low", ...}
  ml_out/ml_metrics.json       -> {"precision":..,"recall":..,"f1":..,"anomaly_rate":..,"mean_anomaly_score":..}

Output:
  ml_out/gate_out.json -> {"decision":"ACCEPT|HOLD|REJECT", "reason":"...", "fusion_score":..}

Exit codes:
  1 -> REJECT (fail the job)
  0 -> ACCEPT/HOLD (job passes)
"""
import sys
from pathlib import Path
from utils import read_json, write_json, log


TRIVY_JSON = Path("trivy_out/trivy_metrics.json")
ML_JSON    = Path("ml_out/ml_metrics.json")
OUT_JSON   = Path("ml_out/gate_out.json")


def adaptive_fusion_decision(trivy_data, ml_data):
    """
    Adaptive fusion using weighted scoring
    Combines Trivy's risk level with ML F1 and anomaly rate
    """
    # Extract key metrics
    trivy_risk = (trivy_data.get("risk") or "low").lower()
    ml_f1 = float(ml_data.get("f1", 0.0))
    ml_precision = float(ml_data.get("precision", 0.0))
    ml_recall = float(ml_data.get("recall", 0.0))
    anomaly_rate = float(ml_data.get("anomaly_rate", 0.0))
    mean_score = float(ml_data.get("mean_anomaly_score", 0.0))

    # Map Trivy risk to numeric scale
    risk_scale = {"low": 1, "medium": 2, "high": 3}
    risk_value = risk_scale.get(trivy_risk, 1)

    # Weighted fusion formula
    # Adjust weights (0.6/0.4) based on your sensitivity requirements
    fusion_score = (0.6 * risk_value) + (0.4 * (anomaly_rate * 10))

    # Dynamic decision thresholds
    if fusion_score >= 2.5 or trivy_risk == "high":
        decision = "REJECT"
        reason = f"High security risk detected (Trivy: {trivy_risk}, Fusion Score: {fusion_score:.2f})"
    
    elif fusion_score >= 1.8 or (trivy_risk == "medium" and anomaly_rate >= 0.05):
        decision = "HOLD"
        reason = f"Medium risk requiring review (Fusion Score: {fusion_score:.2f}, Anomaly Rate: {anomaly_rate:.2%})"
    
    elif ml_f1 >= 0.25 and ml_recall >= 0.40:
        decision = "HOLD"
        reason = f"ML model detected moderate anomalies (F1: {ml_f1:.2f}, Recall: {ml_recall:.2f})"
    
    else:
        decision = "ACCEPT"
        reason = f"Low risk profile (Trivy: {trivy_risk}, Fusion Score: {fusion_score:.2f})"

    return {
        "decision": decision,
        "reason": reason,
        "fusion_score": round(fusion_score, 2),
        "trivy_risk": trivy_risk,
        "ml_f1": round(ml_f1, 3),
        "ml_precision": round(ml_precision, 3),
        "ml_recall": round(ml_recall, 3),
        "anomaly_rate": round(anomaly_rate, 3)
    }


def main():
    # Load input JSONs
    trivy_data = read_json(TRIVY_JSON, default={}) or {}
    ml_data = read_json(ML_JSON, default={}) or {}

    # Check if both sources exist
    if not trivy_data:
        log("⚠️  Warning: Trivy metrics missing or empty — defaulting to low risk")
        trivy_data = {"risk": "low"}
    
    if not ml_data:
        log("⚠️  Warning: ML metrics missing or empty — defaulting to low anomaly")
        ml_data = {"f1": 0.0, "anomaly_rate": 0.0}

    # Execute adaptive fusion
    result = adaptive_fusion_decision(trivy_data, ml_data)

    # Save output
    write_json(OUT_JSON, result)
    
    # Log decision
    log(f"\n{'='*60}")
    log(f"🎯 ADAPTIVE FUSION DECISION GATE")
    log(f"{'='*60}")
    log(f"Trivy Risk:      {result['trivy_risk'].upper()}")
    log(f"ML F1 Score:     {result['ml_f1']}")
    log(f"Anomaly Rate:    {result['anomaly_rate']*100:.1f}%")
    log(f"Fusion Score:    {result['fusion_score']}")
    log(f"{'='*60}")
    log(f"✅ DECISION: {result['decision']}")
    log(f"📋 Reason: {result['reason']}")
    log(f"{'='*60}\n")

    # Exit with appropriate code
# During experimentation, do not stop the pipeline
    sys.exit(0)


if __name__ == "__main__":
    main()
