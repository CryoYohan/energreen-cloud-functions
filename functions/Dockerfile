# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY bridge.py .
COPY isrgrootx1.pem .

# Run the MQTT-to-Pub/Sub bridge script
CMD ["python", "bridge.py"]