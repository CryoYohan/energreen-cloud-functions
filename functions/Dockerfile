# Use a base image with Python 3.11
FROM python:3.11-slim

# Install the system dependency for LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# The command to run the function
CMD ["functions-framework", "--target", "daily_prediction_runner", "--source", "predict.py"]