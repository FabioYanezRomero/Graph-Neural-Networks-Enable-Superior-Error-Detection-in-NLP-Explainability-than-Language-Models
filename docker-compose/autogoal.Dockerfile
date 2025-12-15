# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BUILD_ENV=development \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH=/home/coder/autogoal:/app \
    PATH="/home/coder/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    rsync \
    curl \
    procps \
    lsof \
    net-tools \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Create coder user and set up directories
RUN useradd -m coder && \
    mkdir -p /home/coder/autogoal && \
    chown -R coder:coder /home/coder

# Switch to coder user
USER coder

# Set working directory
WORKDIR /home/coder/autogoal

# Clone the repository
RUN git clone --recursive https://github.com/FabioYanezRomero/autogoal_graph_explainability.git . && \
    git remote add upstream https://github.com/autogoal/autogoal.git

# Copy local changes if they exist
COPY --chown=coder:coder ./autogoal_repo/ /tmp/autogoal_local/
RUN if [ -d "/tmp/autogoal_local" ]; then \
        rsync -a --exclude='.git' /tmp/autogoal_local/ ./ && \
        rm -rf /tmp/autogoal_local; \
    fi

# Install Python tools
RUN pip install --upgrade pip && \
    pip install --user poetry && \
    poetry --version

# Install AutoGOAL in development mode
RUN cd /home/coder/autogoal/autogoal && \
    pip install -e .

# Install autogoal-contrib in development mode if it has a setup file
RUN if [ -d "/home/coder/autogoal/autogoal-contrib" ] && \
       { [ -f "/home/coder/autogoal/autogoal-contrib/setup.py" ] || \
         [ -f "/home/coder/autogoal/autogoal-contrib/pyproject.toml" ]; }; then \
        cd /home/coder/autogoal/autogoal-contrib && \
        touch README.md && \
        pip install -e . --no-cache-dir; \
    elif [ -d "/home/coder/autogoal/autogoal-contrib" ]; then \
        echo "autogoal-contrib found but not installed (missing setup.py/pyproject.toml)"; \
    fi

# Install autogoal-remote in development mode if it has a setup file
RUN if [ -d "/home/coder/autogoal/autogoal-remote" ] && \
       { [ -f "/home/coder/autogoal/autogoal-remote/setup.py" ] || \
         [ -f "/home/coder/autogoal/autogoal-remote/pyproject.toml" ]; }; then \
        cd /home/coder/autogoal/autogoal-remote && \
        touch README.md && \
        pip install -e . --no-cache-dir; \
    elif [ -d "/home/coder/autogoal/autogoal-remote" ]; then \
        echo "autogoal-remote found but not installed (missing setup.py/pyproject.toml)"; \
    fi

# Set the default command
CMD ["/bin/bash"]
