# -----------------------------------------------------------
#  STEP 1: Use older Python base image (medium vulnerabilities)
# -----------------------------------------------------------
FROM python:3.9-slim-bullseye
WORKDIR /app

# -----------------------------------------------------------
#  STEP 2: Install outdated packages + unsafe configs
#  (These intentionally create medium-risk findings)
# -----------------------------------------------------------
RUN apt-get update && \
    apt-get install -y \
      curl=7.74.0-1.3+deb11u11 \
      wget=1.21-1+deb11u1 \
      openssl=1.1.1k-1+deb11u2 && \
    apt-get clean

# -----------------------------------------------------------
#  STEP 3: Inject "fake secrets" (Trivy will flag these)
# -----------------------------------------------------------
RUN echo "AWS_ACCESS_KEY_ID=AKIAFAKE1234567EXAMPLE" >> /app/config.env && \
    echo "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" >> /app/config.env && \
    echo "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----FAKEKEY-----END PRIVATE KEY-----" >> /app/config.env

# -----------------------------------------------------------
#  STEP 4: Misconfiguration — broad permissions
# -----------------------------------------------------------
RUN chmod 777 /app

# -----------------------------------------------------------
#  STEP 5: Example .env file with exposed tokens
# -----------------------------------------------------------
RUN echo "DB_PASSWORD=supersecret123" >> /app/.env && \
    echo "SECRET_TOKEN=sk_live_FAKEKEY123456" >> /app/.env

# -----------------------------------------------------------
#  STEP 6: Application dependencies (old Flask version)
# -----------------------------------------------------------
COPY src/ /app/src
RUN pip install --no-cache-dir Flask==1.0.2  # vulnerable release (CVE-2019 flagged)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "-m", "src.main"]
