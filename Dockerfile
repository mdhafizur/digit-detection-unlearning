# Use a lightweight base image
FROM python:3.12.2-slim

# Set the working directory in the container
WORKDIR /app

# Install build dependencies (including HDF5)
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Expose the port
EXPOSE 5000

# Specify the command to run on container start
CMD ["python3", "app.py"]
