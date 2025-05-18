# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install system dependencies needed by OpenCV, etc.
# And then install Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create directories needed by the app (uploads for temp, data for persistent via volume)
# These directories will be in /app/
RUN mkdir -p uploads data

# Create the resources directory for the class list
RUN mkdir -p resources

# Copy only the necessary application files and assets
COPY app.py .
COPY utils.py .
COPY resources/wlasl_class_list.txt resources/wlasl_class_list.txt
COPY resources/asl_model.pth resources/asl_model.pth
COPY templates templates/
COPY static static/
COPY models/ /app/models/
COPY __init__.py /app/

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python", "app.py"]