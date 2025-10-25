# -------------------------------------------
# Hybrid DevSecOps test – Medium vulnerabilities (safe + compatible)
# -------------------------------------------
FROM python:3.9-slim-bullseye
WORKDIR /app

# Step 1 – Install packages WITHOUT specific version pinning
# (Debian will pull the slightly outdated versions present in repo)
RUN apt-get update && \
    apt-get install -y curl wget openssl && \
    apt-get clean

# Step 2 – Inject dummy secrets for Trivy detection
RUN echo "AWS_ACCESS_KEY_ID=AKIAFAKEMEDIUMKEY" >> /root/.aws && \
    echo "AWS_SECRET_ACCESS_KEY=fake-secret-test-key" >> /root/.aws && \
    echo "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----FAKEKEY-----END PRIVATE KEY-----" >> /root/.keys

# Step 3 – Weak permissions to create config warning
RUN chmod 777 /app

# Step 4 – Minimal fake application file
RUN echo "print('Simulated vulnerable app for Trivy HOLD test')" > /app/app.py

# Step 5 – Outdated Flask to provide package-level CVE
RUN pip install --no-cache-dir Flask==1.0.2

# Step 6 – Non-root mode for realism (keeps risk medium)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "app.py"]
