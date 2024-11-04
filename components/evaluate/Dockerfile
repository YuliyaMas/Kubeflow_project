# Dockerfile for evaluate component
FROM python:3.8-slim

WORKDIR /app

COPY evaluate.py /app/evaluate.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "evaluate.py"]
