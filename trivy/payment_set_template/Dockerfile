FROM python:3.10-slim   # older version – known to have medium CVEs
WORKDIR /app

# intentionally keep outdated packages to trigger Trivy medium risk
RUN apt-get update && \
    apt-get install -y \
    curl=7.74.0-1.3+deb11u10 \
    openssl=1.1.1w-0+deb11u1 \
    wget=1.21-1+deb11u1 && \
    rm -rf /var/lib/apt/lists/*

# simulate weak config flags (Trivy config misconfiguration warning)
RUN chmod 777 /app

# normal non-root setup
RUN useradd -m appuser && chown -R appuser /app
USER appuser

COPY src/ /app/src
RUN pip install --no-cache-dir -r /app/src/requirements.txt

CMD ["python", "-m", "src.main"]
