FROM alpine:3.20
WORKDIR /app

# Keep harmless placeholder environment variables — no sensitive keywords
ENV APP_MODE="demo"
ENV APP_LEVEL="placeholder_value"
ENV DEMO_MODE_ENABLED="true"

# Copy config file (safe)
COPY configs/config.yml /app/config.yml

# Add a simple utility package (minor version pin = medium risk)
RUN apk add --no-cache busybox=1.36.1-r15

# Clean up cache
RUN rm -rf /var/cache/apk/* /tmp/* /var/tmp/*

CMD ["sh", "-c", "echo HOLD stage test executed"]
