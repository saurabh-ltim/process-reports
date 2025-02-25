FROM python:3.10-slim

WORKDIR /app

COPY app.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN curl -sSL https://sdk.cloud.google.com | bash

ENV PATH=$PATH:/root/google-cloud-sdk/bin

EXPOSE 8080

CMD ["python", "app.py"]