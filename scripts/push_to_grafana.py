import os, json, time
from opentelemetry.metrics import get_meter, set_meter_provider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter  # Use gRPC exporter

def jload(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# Load data from previous steps
scan  = jload("trivy_out/scan.json")
tmet  = jload("trivy_out/trivy_metrics.json")
mlmet = jload("ml_out/ml_metrics.json")
gate  = jload("ml_out/gate_out.json")

# Count vulnerability occurrences
def count_vuln(sev):
    c = 0
    for r in scan.get("Results", []):
        for v in r.get("Vulnerabilities", []):
            if v.get("Severity","").upper() == sev:
                c += 1
    return c

# Vulnerability counts per severity
crit, high, med, low = (count_vuln(s) for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
mis  = sum(len((r.get("Misconfigurations") or [])) for r in (scan.get("Results") or []))
secr = sum(len((r.get("Secrets") or [])) for r in (scan.get("Results") or []))

# Machine Learning metrics
prec = float(mlmet.get("precision", 0))
rec  = float(mlmet.get("recall", 0))
f1   = float(mlmet.get("f1", 0))
anom = float(mlmet.get("anomalies_detected", 0))

# Decision Gate metrics
approve = float(gate.get("approve", gate.get("APPROVE", 0)))
hold    = float(gate.get("hold", gate.get("HOLD", 0)))
block   = float(gate.get("block", gate.get("BLOCK", 0)))

repo   = os.getenv("REPO", "unknown")
branch = os.getenv("BRANCH", "unknown")

# OTLP gRPC exporter setup
exporter = OTLPMetricExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))  # Ensure correct endpoint
reader = PeriodicExportingMetricReader(exporter, export_interval_millis=1000)
provider = MeterProvider(
    resource=Resource.create({"service.name": "trivy-ml-ci", "repo": repo, "branch": branch}),
    metric_readers=[reader],
)
set_meter_provider(provider)
meter = get_meter("ci.trivy_ml")

# Counters for the metrics
sev      = meter.create_counter("trivy_severity_count")
findings = meter.create_counter("trivy_findings_total")
gatec    = meter.create_counter("decision_gate_events_total")
anomc    = meter.create_counter("ml_anomalies_detected_total")

# Gauges (one-shot values)
prec_g = meter.create_up_down_counter("ml_precision")
rec_g  = meter.create_up_down_counter("ml_recall")
f1_g   = meter.create_up_down_counter("ml_f1_score")

labels = {"repo": repo, "branch": branch}
for s, val in (("CRITICAL", crit), ("HIGH", high), ("MEDIUM", med), ("LOW", low)):
    sev.add(int(val), dict(labels, severity=s))

# Count findings by type
findings.add(int(mis), dict(labels, type="misconfig"))
findings.add(int(secr), dict(labels, type="secret"))

# Count anomalies detected
anomc.add(int(anom), labels)

# Decision Gate counts
gatec.add(int(approve), dict(labels, action="APPROVE"))
gatec.add(int(hold), dict(labels, action="HOLD"))
gatec.add(int(block), dict(labels, action="BLOCK"))

# Machine learning scores (gauge)
prec_g.add(prec, labels)
rec_g.add(rec, labels)
f1_g.add(f1, labels)

# Give exporter a moment to flush metrics
time.sleep(2)
