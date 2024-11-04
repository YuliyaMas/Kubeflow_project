# Dockerfile for preprocess component
FROM python:3.8-slim

WORKDIR /app

COPY preprocess.py /app/preprocess.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "preprocess.py"]