FROM python:3.9-slim-bullseye
WORKDIR /app

# Step 1 – Add risky older packages & misconfigurations
RUN apt-get update && \
    apt-get install -y \
      curl wget netcat-traditional && \
    apt-get install -y --no-install-recommends openssh-client && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    chmod 777 /app && \
    apt-get clean

# Step 2 – Simulate exposed secrets/configs
RUN echo "AWS_ACCESS_KEY_ID=AKIAEXPOSEDKEY1234" >> /root/.aws && \
    echo "AWS_SECRET_ACCESS_KEY=FAKESECRETKEYABC1234" >> /root/.aws && \
    echo "DB_PASSWORD=admin123" >> /app/config.env && \
    echo "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----FAKEKEY-----END PRIVATE KEY-----" >> /app/keys && \
    chmod 777 /root/.aws /app/config.env /app/keys

# Step 3 – Fake app file to complete build
RUN echo "print('Medium-risk test app')" > /app/app.py

# Step 4 – Install outdated Python dependency (medium CVE trigger)
RUN pip install --no-cache-dir Flask==1.0.2

# Step 5 – Non‑root mode for realism while maintaining medium risk
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "app.py"]
