#!/bin/bash
# Script to run main.py with proper conda environment

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate curvas_aprendizado

# Run the main script  
python main.py

