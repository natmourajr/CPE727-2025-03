#!/bin/bash
# Wrapper script to run baseline evaluation with PyTorch workaround

# Disable CUDA to avoid PyTorch compatibility issues on macOS
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Work around CUDA-related issues on non-CUDA systems
export TORCH_USE_CUDA_DSA=0

# Run the script
uv run python src/evaluate_baseline_models.py "$@"
