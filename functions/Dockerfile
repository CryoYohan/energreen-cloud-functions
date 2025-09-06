# Dockerfile for Cloud Run
FROM python:3.11-slim

# system deps for scikit-learn / pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_nilm.py .

# set port for Cloud Run
ENV PORT 8080

CMD ["uvicorn", "train_nilm:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
