# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter and additional packages
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    matplotlib \
    seaborn \
    scikit-learn \
    pandas \
    tqdm

# Copy the entire project
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 jupyter && \
    chown -R jupyter:jupyter /app
USER jupyter

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=""
ENV JUPYTER_ALLOW_ROOT=0

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
