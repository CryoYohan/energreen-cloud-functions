# Use the official Python image as a base
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Install system-level dependencies required for numerical libraries
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application source code
COPY . .

# Expose the port (if running FastAPI/Uvicorn apps)
EXPOSE 8080

# Allow dynamic command override (train_nilm or cluster_nilm)
ENTRYPOINT ["uvicorn"]
CMD ["cluster_nilm:app", "--host", "0.0.0.0", "--port", "8080"]
