# Use a base image with Python 3.11
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Command to run the function using the functions-framework
# --target is the name of your function
# --source is the name of your Python file
CMD ["functions-framework", "--target", "getApplianceSignatures", "--source", "getApplianceSignatures.py"]