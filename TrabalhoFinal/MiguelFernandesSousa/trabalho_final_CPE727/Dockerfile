# Multi-stage build for Trabalho Final CPE727
FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY run_experiments.py ./

# Install dependencies using uv
RUN uv sync --frozen

# Create directories for outputs
RUN mkdir -p eda/outputs results/plots confusion_matrices_baseline mlruns

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Expose MLflow port
EXPOSE 5000

# Default command (can be overridden)
CMD ["uv", "run", "run_experiments.py", "--help"]
