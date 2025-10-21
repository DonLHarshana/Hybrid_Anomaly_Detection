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

    trivy_risk = (t.get("risk") or "low").lower()
    ml_f1 = float(m.get("f1", 0))
    anomaly_rate = float(m.get("anomaly_rate", 0))
    mean_score = float(m.get("mean_anomaly_score", 0))

    # --- Policy (adjust as needed) ---
    if trivy_risk == "high":
        decision, reason = "REJECT", "Trivy risk is HIGH"

    elif ml_f1 >= 0.60 and anomaly_rate >= 0.05:
        decision, reason = "REJECT", "ML flags significant anomalies (F1 >= 0.60 and anomaly_rate >= 5%)"

    elif trivy_risk == "medium" or anomaly_rate >= 0.02 or mean_score >= 0.0:  # tweak mean_score cutoff per data
        decision, reason = "HOLD", "Medium risk or moderate anomaly signal"

    else:
        decision, reason = "ACCEPT", "Low risk and low anomaly rate"

    write_json(OUT_JSON, {"decision": decision, "reason": reason})
    log(f"Decision: {decision} — {reason}")

    sys.exit(1 if decision == "REJECT" else 0)

if __name__ == "__main__":
    main()
