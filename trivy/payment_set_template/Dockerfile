# -------------------------------------------------
# ✅ Guaranteed Medium‑risk build for HOLD scenario
# -------------------------------------------------
FROM python:3.8-slim-buster
WORKDIR /app

# Step 1. Vulnerable base & outdated libs
RUN apt-get update && \
    apt-get install -y curl wget openssh-client openssl && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    apt-get clean

# Step 2. Inject exposed secrets (Trivy secret + config detector)
RUN echo "AWS_ACCESS_KEY_ID=AKIAFAKETRIVYMEDIUMTEST" >> /app/config.env && \
    echo "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/fakekey" >> /app/config.env && \
    echo "DB_PASSWORD=admin123" >> /app/config.env && \
    echo "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----FAKEKEYDATA-----END PRIVATE KEY-----" >> /app/config.env

# Step 3. Misconfiguration (broad permissions)
RUN chmod -R 777 /app

# Step 4. Add dummy vulnerable Python app
RUN echo "print('Simulated app for HOLD test')" > /app/app.py

# Step 5. OLD Flask (CVE‑2019‑Critical → Medium)
RUN pip install --no-cache-dir Flask==1.0.2

# Step 6. Optional: Keeping danger moderate (no root runtime)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "app.py"]
