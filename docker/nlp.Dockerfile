# Base image with Python 3.9
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install only the required Python packages
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    scipy>=1.7.0 \
    scikit-learn>=1.0.0 \
    tqdm>=4.65.0 \
    pyyaml>=5.4.1 \
    stanza>=1.5.0 \
    datasets>=2.12.0 \
    peft>=0.6.0 \
    transformers>=4.30.0

# Install AllenNLP with specific versions to avoid conflicts
RUN pip install --no-cache-dir \
    allennlp==2.10.1 \
    allennlp-models==2.10.1

# Skip checklist installation as it's causing issues
# RUN pip install --no-cache-dir checklist

# Copy the rest of the application code
COPY . .

# Set the working directory to src where the main code is located
WORKDIR /app/src

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command to keep the container running
CMD ["bash"]
