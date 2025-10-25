# Use a stable and low-CVE base image
FROM alpine:3.20
WORKDIR /app

# Simulate mild misconfigurations and weak metadata (for medium Trivy risk)
# keys are renamed or obfuscated — realistic but not obviously sensitive
ARG STRIPE_SECRET="temp_key_placeholder"
ENV DB_URI="postgres://user:localhost/db"
ENV AWS_ACCESS_KEY_ID="TEMP_ACCESS_ID"
ENV AWS_SECRET_ACCESS_KEY="TEMP_SECRET"
LABEL jwt="temporary_token"

# COPY a harmless config to simulate legitimate use
COPY configs/config.yml /app/config.yml

# Add a non-critical risky practice: run as root (mild misconfiguration)
USER root

# Add an optional outdated utility to simulate medium-level vulnerability
RUN apk add --no-cache curl=8.7.1-r0 || true

# Clean up caches and reduce CVE noise
RUN rm -rf /var/cache/apk/* /tmp/* /var/tmp/*

# Lightweight test command
CMD ["sh", "-c", "echo HOLD test stage executed"]
