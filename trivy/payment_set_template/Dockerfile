FROM alpine:3.20
WORKDIR /app

# Keep minor misconfigurations to simulate medium risk
ENV APP_MODE="test"
ENV TEMP_TOKEN="temporary_value"

# Copy only safe config (no sensitive env variables or files)
COPY configs/config.yml /app/config.yml

# Optional: install a safe package - remove curl entirely (Trivy flags it)
# RUN apk add --no-cache curl=8.7.1-r0
# Instead, simulate a minor OS risk by pinning an old core utility:
RUN apk add --no-cache busybox-suid

# Clean up and remove traces
RUN rm -rf /var/cache/apk/* /tmp/* /var/tmp/*

CMD ["sh", "-c", "echo HOLD stage test executed"]
