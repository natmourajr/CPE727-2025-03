ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04

FROM $BASE_IMAGE

# Set timezone
ENV TZ=Etc/UTC

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

# Copy entire repository (needed for local dependencies)
COPY . /app

WORKDIR /app/src/experiments/WineMLPExperiment

# Setup UV and Python environment
RUN uv venv /venv
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV UV_PYTHON=/venv/bin/python
ENV PATH=/venv/bin:$PATH

# Install dependencies (including local packages)
RUN uv sync --locked

# Default command: run the experiment
CMD ["wine-mlp-experiment", "--epochs", "50"]
