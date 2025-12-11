#!/bin/bash
# Run experiments with correct environment

# Unset problematic DYLD variables that interfere with PyTorch
unset DYLD_LIBRARY_PATH
unset DYLD_FALLBACK_LIBRARY_PATH
unset DYLD_INSERT_LIBRARIES

# Run the experiments
uv run python run_all_experiments.py "$@"
