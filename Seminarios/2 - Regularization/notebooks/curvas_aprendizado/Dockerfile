FROM python:3.10-slim as base

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (using BuildKit cache for faster rebuilds)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p results data

# Run the training
CMD ["python", "main.py"]

