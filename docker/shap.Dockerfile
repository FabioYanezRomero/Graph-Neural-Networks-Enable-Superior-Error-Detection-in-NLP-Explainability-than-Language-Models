# Dedicated Dockerfile for SHAP
FROM ubuntu:22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    build-essential git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set workdir
WORKDIR /shap

# (Optional) Copy local project files. Remove if not needed.
# COPY . /shap

# Upgrade pip and install SHAP (from PyPI)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install shap

# Optional: install jupyter for interactive use
RUN python3 -m pip install jupyter

# Set default entrypoint (bash shell)
ENTRYPOINT ["/bin/bash"]