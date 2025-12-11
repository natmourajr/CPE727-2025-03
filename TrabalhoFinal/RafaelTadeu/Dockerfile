# Multi-stage Dockerfile for EuroSAT DL Comparison Project
# Optimized for PyTorch with CUDA support

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Stage 3: Application
FROM base AS application

# Copy installed dependencies from previous stage
COPY --from=dependencies /app/.venv /app/.venv

# Copy project files
COPY . .

# Set Python path to use virtual environment
ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/app/.venv"

# Create necessary directories
RUN mkdir -p data/01_raw data/02_intermediate data/03_primary \
    data/04_feature data/05_model_input data/06_models \
    data/07_model_output data/08_reporting logs

# Expose ports for Kedro Viz and Tensorboard
EXPOSE 4141 6006

# Default command to run Kedro
CMD ["kedro", "run"]

# Optional: Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1
