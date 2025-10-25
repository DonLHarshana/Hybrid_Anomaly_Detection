FROM python:3.12-slim
RUN apt-get update && apt-get install -y curl
WORKDIR /app


ARG STRIPE_SECRET={{GENERIC_API_KEY}}
ENV DB_URI={{POSTGRES_URI}}
ENV AWS_ACCESS_KEY_ID={{AWS_ACCESS_KEY_ID}}
ENV AWS_SECRET_ACCESS_KEY={{AWS_SECRET_ACCESS_KEY}}
LABEL jwt="{{DUMMY_JWT}}"
# Principle of least privilege - non‑root execution
RUN useradd -m appuser
USER appuser

# Copy only necessary app files (avoid secrets)
COPY src/ /app/src

# Do NOT hardcode ARG/ENV secrets
# Instead, use runtime environment injection via GitHub Secrets
# ENV JWT=""  # will be set dynamically in CI/CD

# Install limited dependencies
RUN pip install --no-cache-dir -r /app/src/requirements.txt

# Final command
CMD ["python", "-m", "src.main"]
