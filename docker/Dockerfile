# Use a CUDA 12.1 runtime so we only need NVIDIA drivers >= 530 on the host
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Madrid \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH}

# Add LLVM repository for OpenMP packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    procps \
    lsof \
    net-tools \
    iproute2 \
    build-essential \
    ca-certificates \
    libopenblas-dev \
    libomp-dev \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Ensure latest pip tooling
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.3.1 with CUDA 12.1 (compatible with driver >= 530)
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric with wheels matching PyTorch 2.3.1 + cu121
RUN pip install --no-cache-dir \
    pyg_lib==0.4.0+pt23cu121 \
    torch_scatter==2.1.2+pt23cu121 \
    torch_sparse==0.6.18+pt23cu121 \
    torch_cluster==1.6.3+pt23cu121 \
    torch_spline_conv==1.2.2+pt23cu121 \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

# Install torch-geometric itself after the binary wheels are present
RUN pip install --no-cache-dir torch_geometric==2.5.3

# Install remaining dependencies (excluding torch and geometric packages)
COPY requirements.txt .
RUN grep -v -E 'torch|pytorch_geometric|pyg_lib|torch_scatter|torch_sparse|torch_cluster|torch_spline_conv' requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    rm requirements.txt requirements_filtered.txt

# Copy the rest of the application code
COPY . .

# Set working directory for main code
WORKDIR /app/src

# Create a non-root user with video/render access and ensure helper scripts are executable
RUN groupadd -f video \
    && groupadd -f render \
    && useradd -m appuser \
    && usermod -aG video,render appuser \
    && chown -R appuser:appuser /app \
    && find /app/scripts -type f -name '*.sh' -exec chmod +x {} +
USER appuser

# Default command to keep the container running
CMD ["bash"]
