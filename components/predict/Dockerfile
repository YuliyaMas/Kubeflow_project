# Dockerfile for predict component
FROM python:3.8-slim

WORKDIR /app

COPY predict.py /app/generate.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "generate.py"]
