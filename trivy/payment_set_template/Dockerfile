FROM alpine:3.20
WORKDIR /app

# anti-patterns ON PURPOSE (fake secrets) so Trivy has targets
ARG STRIPE_SECRET={{GENERIC_API_KEY}}
ENV DB_URI={{POSTGRES_URI}}
ENV AWS_ACCESS_KEY_ID={{AWS_ACCESS_KEY_ID}}
ENV AWS_SECRET_ACCESS_KEY={{AWS_SECRET_ACCESS_KEY}}
LABEL jwt="{{DUMMY_JWT}}"

# also copy .env (bad practice, just for testing)
COPY configs/.env /app/.env

CMD ["sh","-c","echo ok"]
