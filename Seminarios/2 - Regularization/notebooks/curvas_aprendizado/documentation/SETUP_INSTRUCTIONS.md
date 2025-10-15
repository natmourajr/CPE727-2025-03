# PyTorch Setup for Mac Silicon - Complete Guide

## The Problem
The Python ecosystem on macOS has compatibility issues between:
- Conda/Miniconda Python builds
- PyTorch precompiled binaries
- Mac Silicon (ARM64) architecture

## SOLUTION: Use System Python or Pyenv

### Option 1: System Python with venv (RECOMMENDED)

```bash
# Check your system Python version
python3 --version

# It should be 3.9, 3.10, or 3.11 (NOT 3.12 or 3.13)

# If you don't have a compatible version, install with Homebrew:
brew install python@3.10

# Create venv with system Python
python3.10 -m venv .venv

# Activate
source .venv/bin/activate

# Install PyTorch (this will get the correct ARM64 build)
pip install torch torchvision

# Install other dependencies
pip install numpy matplotlib tqdm

# Test
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"

# Run your script
python main.py
```

### Option 2: Use Miniforge (Better than Miniconda for ARM64)

```bash
# Install Miniforge if you don't have it
brew install miniforge

# Initialize
conda init zsh

# Restart terminal

# Create environment with explicit ARM64 support
CONDA_SUBDIR=osx-arm64 conda create -n curvas python=3.10 -y
conda activate curvas

# Lock environment to ARM64
conda config --env --set subdir osx-arm64

# Install PyTorch via pip (more reliable than conda)
pip install torch torchvision numpy matplotlib tqdm

# Test
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"

# Run
python main.py
```

### Option 3: Docker (Most Reliable)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install torch torchvision numpy matplotlib tqdm

COPY . .
CMD ["python", "main.py"]
```

Run:
```bash
docker build -t cifar10-training .
docker run -v $(pwd)/results:/app/results cifar10-training
```

## Why This Happens

1. **Miniconda Python** is built with specific symbols that don't match PyTorch's expectations
2. **PyTorch precompiled wheels** expect standard CPython symbols
3. **ARM64 detection** sometimes fails, leading to wrong architecture binaries

## The Fix You Mentioned

Your suggestion to use `conda config --env --set subdir osx-arm64` is **correct** and part of the solution, but conda's Python itself has compatibility issues with PyTorch.

**The best approach**: Use system Python (via Homebrew) with venv, or use Miniforge instead of Miniconda.

## Quick Test

After setup, always test:
```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

You should see:
```
2.x.x
True
```

## Current Status

Your project code (`main.py`) is already fixed to support MPS. You just need a working Python + PyTorch installation.

