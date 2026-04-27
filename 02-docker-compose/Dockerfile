FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
    ray[default] \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    ipython

# Create a non-root user (optional but recommended)
RUN useradd -m -s /bin/bash ray

# Expose ports
EXPOSE 6379 8265 10001

# Default command (can be overridden by docker-compose)
CMD ["ray", "start", "--head"]
