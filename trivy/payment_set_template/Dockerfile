FROM python:3.11  # slightly older base image → known CVEs
WORKDIR /app

# intentionally installing outdated packages
RUN apt-get update && \
    apt-get install -y wget=1.21-1+deb11u1 curl=7.74.0-1.3+deb11u11 && \
    apt-get clean

# simulate configuration mispractice (Trivy will flag chmod 777)
RUN chmod -R 777 /app

# normal user setup for isolation
RUN useradd -m appuser && chown -R appuser /app
USER appuser

COPY src/ /app/src
RUN pip install --no-cache-dir -r /app/src/requirements.txt

CMD ["python", "-m", "src.main"]
