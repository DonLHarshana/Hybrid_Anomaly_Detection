FROM python:3.12-slim
WORKDIR /app

# deliberately install one extra package (creates medium vulnerability)
RUN apt-get update && apt-get install -y curl

# non-root user for better isolation
RUN useradd -m appuser && chown -R appuser /app
USER appuser

COPY src/ /app/src
RUN pip install --no-cache-dir -r /app/src/requirements.txt

CMD ["python", "-m", "src.main"]
