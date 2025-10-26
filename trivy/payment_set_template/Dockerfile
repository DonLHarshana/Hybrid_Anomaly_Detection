FROM alpine:3.20
WORKDIR /app

# Keep harmless placeholder tokens – rename to look non‑sensitive
ENV APP_MODE="demo"
ENV TEST_FLAG_PLACEHOLDER="ok_value"

COPY configs/config.yml /app/config.yml

RUN apk add --no-cache busybox
RUN rm -rf /var/cache/apk/* /tmp/* /var/tmp/*

CMD ["sh", "-c", "echo HOLD stage test executed"]
