# scripts/push_to_grafana.py
import os, json, time
from opentelemetry.metrics import get_meter, set_meter_provider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

def jload(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

scan  = jload("trivy_out/scan.json")
tmet  = jload("trivy_out/trivy_metrics.json")
mlmet = jload("ml_out/ml_metrics.json")
gate  = jload("ml_out/gate_out.json")

def count_vuln(sev):
    c = 0
    for r in scan.get("Results", []) or []:
        for v in r.get("Vulnerabilities", []) or []:
            if (v.get("Severity","").upper() == sev):
                c += 1
    return c

crit, high, med, low = (count_vuln(s) for s in ("CRITICAL","HIGH","MEDIUM","LOW"))
mis  = sum(len((r.get("Misconfigurations") or [])) for r in (scan.get("Results") or []))
secr = sum(len((r.get("Secrets")            or [])) for r in (scan.get("Results") or []))

prec = float(mlmet.get("precision", 0))
rec  = float(mlmet.get("recall", 0))
f1   = float(mlmet.get("f1", 0))
anom = float(mlmet.get("anomalies_detected", 0))

approve = float(gate.get("approve", gate.get("APPROVE", 0)))
hold    = float(gate.get("hold",    gate.get("HOLD",    0)))
block   = float(gate.get("block",   gate.get("BLOCK",   0)))

repo   = os.getenv("REPO","unknown")
branch = os.getenv("BRANCH","unknown")

# Uses env: OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_PROTOCOL / OTEL_EXPORTER_OTLP_HEADERS
exporter = OTLPMetricExporter()
reader   = PeriodicExportingMetricReader(exporter, export_interval_millis=1000)
provider = MeterProvider(
    resource=Resource.create({"service.name":"trivy-ml-ci","repo":repo,"branch":branch}),
    metric_readers=[reader],
)
set_meter_provider(provider)
meter = get_meter("ci.trivy_ml")

# Counters
sev      = meter.create_counter("trivy_severity_count")
findings = meter.create_counter("trivy_findings_total")
gatec    = meter.create_counter("decision_gate_events_total")
anomc    = meter.create_counter("ml_anomalies_detected_total")

# Gauges via up/down counters (one-shot value)
prec_g = meter.create_up_down_counter("ml_precision")
rec_g  = meter.create_up_down_counter("ml_recall")
f1_g   = meter.create_up_down_counter("ml_f1_score")

labels = {"repo":repo,"branch":branch}
for s,val in (("CRITICAL",crit),("HIGH",high),("MEDIUM",med),("LOW",low)):
    sev.add(int(val), dict(labels, severity=s))

findings.add(int(mis),  dict(labels, type="misconfig"))
findings.add(int(secr), dict(labels, type="secret"))

anomc.add(int(anom), labels)

gatec.add(int(approve), dict(labels, action="APPROVE"))
gatec.add(int(hold),    dict(labels, action="HOLD"))
gatec.add(int(block),   dict(labels, action="BLOCK"))

prec_g.add(prec, labels); rec_g.add(rec, labels); f1_g.add(f1, labels)

# Give the exporter a moment to flush
time.sleep(2)
