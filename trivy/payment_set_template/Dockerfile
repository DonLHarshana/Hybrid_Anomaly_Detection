# ================================
# Step 1: Use slightly older base
# ================================
FROM python:3.9-slim-bullseye
WORKDIR /app

# =========================================
# Step 2: Force Trivy to detect vulnerabilities
# =========================================
# inject out-of-date packages known to trigger medium CVEs
RUN apt-get update && \
    apt-get install -y \
      wget=1.21-1+deb11u1 \
      curl=7.74.0-1.3+deb11u11 \
      openssl=1.1.1k-1+deb11u2 && \
    apt-get clean

# =========================================
# Step 3: Inject dummy “secrets” for Trivy
# =========================================
# These values trigger medium‑severity “hard‑coded credential” warnings.
RUN echo "AWS_ACCESS_KEY_ID=AKIAFAKEMEDIUMKEY" >> /app/env.txt && \
    echo "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/FAKEKEY/CYEXAMPLE" >> /app/env.txt && \
    echo "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----FAKE-----END PRIVATE KEY-----" >> /app/keys.txt

# =========================================
# Step 4: Insecure file permissions
# =========================================
RUN chmod -R 777 /app

# =========================================
# Step 5: Copy and dependencies
# =========================================
COPY src/ /app/src
RUN pip install --no-cache-dir Flask==1.0.2  # explicitly vulnerable version (CVE‑2019)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "-m", "src.main"]
