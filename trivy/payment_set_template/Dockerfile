FROM python:3.9-slim-bullseye
WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
      curl=7.74.0-1.3+deb11u11 \
      wget=1.21-1+deb11u1 \
      openssl=1.1.1k-1+deb11u2 && \
    apt-get clean

RUN echo "AWS_ACCESS_KEY_ID=AKIAFAKEMEDIUMKEY" >> /root/.aws && \
    echo "AWS_SECRET_ACCESS_KEY=fake-secret-test-key" >> /root/.aws && \
    echo "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----FAKEKEY-----END PRIVATE KEY-----" >> /root/.keys && \
    chmod 777 /app

COPY src/ /app/src
RUN pip install --no-cache-dir Flask==1.0.2
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "-m", "src.main"]
