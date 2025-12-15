# Dedicated Dockerfile for TokenSHAP (token-level SHAP for LLMs) with GPU support.
# Use CUDA 12.1 to keep the driver requirement aligned with the main container.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH} \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    build-essential git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /tokenshap

# Upgrade pip/setuptools/wheel first
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.1 build) + torchvision + torchaudio
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Transformers, Datasets, and related NLP utilities
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    huggingface_hub \
    sentencepiece

# Install TokenSHAP from PyPI
RUN pip install --no-cache-dir tokenshap

# Copy and install application requirements
COPY requirements.txt /app/requirements.txt
RUN grep -v -E 'torch|pytorch' /app/requirements.txt > /tmp/requirements-tokenshap.txt && \
    pip install --no-cache-dir -r /tmp/requirements-tokenshap.txt && \
    rm /tmp/requirements-tokenshap.txt

# Optional: Jupyter for interactive development
RUN pip install --no-cache-dir jupyter

# Pre-create cache directories and set permissions
RUN mkdir -p /opt/hf_cache && chmod -R 777 /opt/hf_cache


ENTRYPOINT ["/bin/bash"]
