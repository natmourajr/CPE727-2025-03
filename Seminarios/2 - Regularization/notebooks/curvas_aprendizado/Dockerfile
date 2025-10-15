FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (using precompiled wheels, no compilation needed)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision numpy matplotlib tqdm

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p results data

# Run the training
CMD ["python", "main.py"]

