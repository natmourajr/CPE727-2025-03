ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04

FROM $BASE_IMAGE

ARG EXPERIMENT

# Set timezone
ENV TZ=Etc/UTC

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

# Copy the entire src directory to get dataloaders and experiments
COPY . /app

WORKDIR /app/experiments/${EXPERIMENT}

# Setup UV and Python environment
# Let uv install Python automatically
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13

# Create and activate virtual environment
RUN uv venv

# Install dependencies including the project itself
# uv sync installs the project and all dependencies (including editable ones)
RUN if [ -f uv.lock ]; then \
        uv sync --locked; \
    else \
        uv pip install -e .; \
    fi

# Set up environment to use the venv by default
# These must be set so the venv is active for all commands
ENV VIRTUAL_ENV=/app/experiments/${EXPERIMENT}/.venv
ENV PATH=/app/experiments/${EXPERIMENT}/.venv/bin:$PATH


# Tell uv to always use the project environment we created
# This prevents uv run from trying to create a new environment
ENV UV_NO_SYNC=1

# Create necessary data directories
RUN mkdir -p /app/Data/MNIST /app/Data/NoisyMnist /app/Data/autoencoder/splits /app/Data/autoencoder/params

# Default command - can be overridden at runtime
CMD ["/bin/bash"]
