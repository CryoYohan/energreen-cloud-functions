# Use the official Python image as a base
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application source code
COPY . .

# Expose the port that the application will run on
EXPOSE 8080

# The command to run your application using Uvicorn
# Uvicorn will listen on 0.0.0.0, allowing traffic from the outside
# The `predict_nilm:app` specifies the module `predict_nilm` and the FastAPI app object `app`
CMD ["uvicorn", "predict_nilm:app", "--host", "0.0.0.0", "--port", "8080"]