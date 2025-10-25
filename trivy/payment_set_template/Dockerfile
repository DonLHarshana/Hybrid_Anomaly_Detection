# -----------------------------------------------------------
# Hybrid DevSecOps HOLD Test – Medium Vulnerability Injection
# -----------------------------------------------------------

FROM python:3.9-slim-bullseye
WORKDIR /app

# Step 1 – Install intentionally outdated packages
RUN apt-get update && \
    apt-get install -y \
      curl=7.74.0-1.3+deb11u11 \
      wget=1.21-1+deb11u1 \
      openssl=1.1.1k-1+deb11u2 && \
    apt-get clean

# Step 2 – Inject synthetic secrets (flagged by Trivy)
RUN echo "AWS_ACCESS_KEY_ID=AKIAFAKEMEDIUMKEY" >> /root/.aws && \
    echo "AWS_SECRET_ACCESS_KEY=fake-secret-test-key" >> /root/.aws && \
    echo "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----FAKEKEY-----END PRIVATE KEY-----" >> /root/.keys

# Step 3 – Insecure permissions (config misconfiguration)
RUN chmod 777 /app

# Step 4 – Dummy vulnerable file (replaces missing src folder)
RUN echo "print('Simulated vulnerable app layer')" > /app/app.py

# Step 5 – Install outdated Flask (contains CVE-2019 vulnerabilities)
RUN pip install --no-cache-dir Flask==1.0.2

# Step 6 – Add a non‑root user to avoid “critical” escalation
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "app.py"]
