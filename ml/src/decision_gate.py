"""
Compares Trivy risk + ML signals and decides: ACCEPT / HOLD / REJECT

Inputs:
  trivy_out/trivy_metrics.json -> {"precision":..,"recall":..,"f1":..,"risk":"high|medium|low", ...}
  ml_out/ml_metrics.json       -> {"precision":..,"recall":..,"f1":..,"anomaly_rate":..,"mean_anomaly_score":..}

Output:
  ml_out/gate_out.json -> {"decision":"ACCEPT|HOLD|REJECT","reason":"..."}

Exit codes:
  1 -> REJECT (fail the job)
  0 -> ACCEPT/HOLD (job passes; you can switch HOLD to fail if desired)
"""
import sys
from pathlib import Path
from utils import read_json, write_json, log

TRIVY_JSON = Path("trivy_out/trivy_metrics.json")
ML_JSON    = Path("ml_out/ml_metrics.json")
OUT_JSON   = Path("ml_out/gate_out.json")

def main():
    t = read_json(TRIVY_JSON, default={}) or {}
    m = read_json(ML_JSON, default={}) or {}

    # --- Policy (avoid hard reject when no true positives) ---
    trivy_risk = (t.get("risk") or "low").lower()
    trivy_tp = int(t.get("TP") or 0)
    trivy_score = float(t.get("trivy_risk_score") or 0.0)

    if trivy_risk == "high" and trivy_tp > 0 and trivy_score >= 3.0:
        decision, reason = "REJECT", "Trivy risk is HIGH with confirmed true positives"

    elif trivy_risk == "high" and trivy_tp == 0:
        decision, reason = "HOLD", "Trivy high but 0 true positives (likely noisy; needs review)"

    elif ml_f1 >= 0.60 and anomaly_rate >= 0.05:
        decision, reason = "REJECT", "ML flags significant anomalies (F1 >= 0.60 and anomaly_rate >= 5%)"

    elif trivy_risk == "medium" or anomaly_rate >= 0.02:
        decision, reason = "HOLD", "Medium risk or moderate anomaly signal"

    else:
        decision, reason = "ACCEPT", "Low risk and low anomaly rate"


    write_json(OUT_JSON, {"decision": decision, "reason": reason})
    log(f"Decision: {decision} — {reason}")

    sys.exit(0)

if __name__ == "__main__":
    main()
