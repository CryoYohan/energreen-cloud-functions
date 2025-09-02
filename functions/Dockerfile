# Use a base image with Python 3.11
FROM python:3.11-slim

# Install the system dependency for the cryptography library
RUN apt-get update && apt-get install -y libgomp1

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# The command to run the function
# Note: We are targeting the token_generator_function.py file and the function inside it.
CMD ["functions-framework", "--target", "token_generator_function", "--source", "token_generator_function.py"]
