# PyTorch on macOS Sequoia 15 (26.0) - Complete Solution

## The Problem

You're on **macOS Sequoia 15.0** which has Python C API symbol changes that break current PyTorch precompiled wheels. This affects all Python versions and installation methods.

**Error:** `symbol not found in flat namespace '_PyCode_GetVarnames'` (or similar)

This is NOT your fault - it's an ecosystem incompatibility between:
- macOS 26.0 (Sequoia 15)
- Homebrew Python builds  
- PyTorch precompiled ARM64 wheels

## Solutions (in order of recommendation)

### ✅ Solution 1: Build PyTorch from Source (Works but slow)

```bash
# Activate your venv
source .venv/bin/activate

# Install build dependencies
pip install cmake ninja

# Build PyTorch from source (takes 30-60 minutes)
pip uninstall torch torchvision -y
pip install --no-binary :all: --compile torch torchvision

# This compiles PyTorch locally against your Python version
```

### ✅ Solution 2: Use Nightly Builds (Faster, may have bugs)

```bash
source .venv/bin/activate

pip uninstall torch torchvision -y

# Install PyTorch nightly (built more recently)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Test
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

### ✅ Solution 3: Use Docker (Most Reliable)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install torch torchvision numpy matplotlib tqdm

# Copy project
COPY . .

CMD ["python", "main.py"]
```

Run:
```bash
docker build -t cifar10-training .
docker run -v $(pwd)/results:/app/results -v $(pwd)/data:/app/data cifar10-training
```

### ✅ Solution 4: Use Google Colab (Zero Setup)

Upload your code to Google Colab - it has PyTorch pre-installed:

```python
# In Colab notebook
!git clone <your-repo-url>
%cd curvas_aprendizado
!pip install -r requirements.txt
!python main.py
```

### ⚠️ Solution 5: Downgrade Python (Not recommended)

The issue affects all current Python/Homebrew combinations on Sequoia.

## Recommended Action

**Try Solution 2 (Nightly Builds) first - it's fastest:**

```bash
cd /Users/miguel/Developer/msc/disciplinas/deeplearning/CPE727-2025-03/Seminarios/2\ -\ Regularization/notebooks/curvas_aprendizado
source .venv/bin/activate
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
python -c "import torch; print('✅ PyTorch:', torch.__version__); print('✅ MPS:', torch.backends.mps.is_available())"
python main.py
```

If that fails, use **Docker (Solution 3)** - it's isolated from macOS issues.

## Your Code Status

✅ `main.py` - Already fixed to support MPS  
✅ `requirements.txt` - Updated  
✅ Project structure - Good to go

**The only issue is the PyTorch installation on macOS Sequoia.**

## Future Prevention

This won't be an issue once:
1. PyTorch releases wheels compiled against newer Python C API
2. Or you use Docker/Colab for training

## Why This Happened

macOS Sequoia 15 (26.0) is very new and introduced Python C API changes. PyTorch's precompiled wheels were built before these changes, causing symbol mismatches.

Your `conda config --env --set subdir osx-arm64` suggestion was **correct** - but it doesn't fix this deeper OS-level incompatibility.

---

**TL;DR:** Run the command block under "Recommended Action" to try PyTorch nightly, or use Docker for guaranteed compatibility.

