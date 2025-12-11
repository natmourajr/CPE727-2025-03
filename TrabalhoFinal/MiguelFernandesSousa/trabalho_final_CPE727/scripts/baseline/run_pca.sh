#!/bin/bash
# Wrapper script to run PCA experiments with PyTorch workaround

# Disable CUDA to avoid PyTorch 2.9.1 + Python 3.13 import bug
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the script
uv run python src/train_with_pca.py "$@"
