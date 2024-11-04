# Dockerfile for train component
FROM python:3.8-slim

WORKDIR /app

COPY train.py /app/train.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "train.py"]
