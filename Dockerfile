# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of the application files into the container
COPY requirements.txt .

# Install Python dependencies with no cache for a smaller image size
RUN pip install --no-cache-dir -r requirements.txt

