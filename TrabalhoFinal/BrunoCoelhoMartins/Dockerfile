# ==========================================================
# 1. Base image (Python + Linux)
# ==========================================================
FROM python:3.11-slim

# Avoid Python buffering
ENV PYTHONUNBUFFERED=1

# ==========================================================
# 2. Install system dependencies
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# 3. Create working folder
# ==========================================================
WORKDIR /app

# ==========================================================
# 4. Copy only requirements first (for caching)
# ==========================================================
COPY requirements.txt .

# ==========================================================
# 5. Install Python dependencies
# ==========================================================
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# ==========================================================
# 6. Copy full project (including your local 3W package)
# ==========================================================
COPY . .

# ==========================================================
# 7. Install your LOCAL 3W package
# ==========================================================
RUN pip install -e ./3W

# ==========================================================
# 8. Default command
# ==========================================================
CMD ["python", "script.py"]
