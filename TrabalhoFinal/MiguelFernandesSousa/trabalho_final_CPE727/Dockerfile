# Multi-stage Dockerfile for IARA Deep Learning Project
# Supports both CPU and GPU (CUDA) builds

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.1.0
ARG CUDNN_VERSION=8

# Base stage with uv
FROM python:${PYTHON_VERSION}-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# CPU-only stage
FROM base as cpu

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Install dependencies with uv (CPU version)
# Install PyTorch separately to avoid index conflicts
RUN uv pip install --system numpy scipy pandas scikit-learn librosa soundfile matplotlib seaborn tqdm && \
    uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system -e ".[dev,experiment]"

# GPU (CUDA) stage
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu22.04 as gpu-base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda

FROM gpu-base as gpu

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Install dependencies with uv (GPU version)
RUN uv pip install --system numpy scipy pandas scikit-learn librosa soundfile matplotlib seaborn tqdm && \
    uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install --system -e ".[dev,experiment]"

# Development stage (includes Jupyter and all extras)
FROM cpu as dev

# Install additional dev dependencies
RUN uv pip install --system -e ".[dev,experiment,notebook]"

# Expose Jupyter port
EXPOSE 8888

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash iara && \
    chown -R iara:iara /workspace

USER iara

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage (minimal)
FROM cpu as prod

# Copy only necessary files
COPY test_dataset.py ./

# Run as non-root
RUN useradd -m -u 1000 -s /bin/bash iara && \
    chown -R iara:iara /workspace

USER iara

CMD ["python", "-m", "pytest", "tests/"]
