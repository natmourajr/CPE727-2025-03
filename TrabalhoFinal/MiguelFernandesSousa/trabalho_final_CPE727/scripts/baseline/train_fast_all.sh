#!/bin/bash

# Fast Training Script - All Fashion MNIST Models
# Strategy: Stage 1 (tiny grid search) + Final (full data, 1 epoch)
# Estimated time: ~114 minutes for all 4 models

echo ""
echo "=========================================================================="
echo "FAST TRAINING - ALL 4 FASHION MNIST MODELS"
echo "=========================================================================="
echo ""
echo "Strategy:"
echo "  1. Stage 1: Grid search on 500 samples (find best hyperparameters)"
echo "  2. Final: Train on 60,000 samples with 1 EPOCH ONLY"
echo ""
echo "Estimated time per model:"
echo "  - CNN:       ~28 minutes (5min stage1 + 23min final)"
echo "  - LeNet:     ~33 minutes (5min stage1 + 28min final)"
echo "  - MobileNet: ~53 minutes (5min stage1 + 48min final)"
echo "  - ResNet:    ~89 minutes (5min stage1 + 84min final)"
echo ""
echo "Total estimated time: ~203 minutes (~3.4 hours)"
echo ""
echo "Note: Training with only 1 epoch means models will be undertrained"
echo "      Expected accuracy: ~70-85% (vs 91-95% with full training)"
echo ""
echo "=========================================================================="
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

# Track total time
start_time=$(date +%s)

# Train all models
uv run python src/train_deep_fast.py --dataset fashion_mnist --all

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
total_minutes=$((total_time / 60))
total_seconds=$((total_time % 60))

echo ""
echo "=========================================================================="
echo "TRAINING COMPLETE!"
echo "=========================================================================="
echo ""
echo "Total time: ${total_minutes}m ${total_seconds}s"
echo ""
echo "Next steps:"
echo "  1. View results: mlflow ui"
echo "  2. Open: http://localhost:5000"
echo "  3. Compare experiments:"
echo "     - fashion_mnist_cnn_final_1epoch"
echo "     - fashion_mnist_lenet_final_1epoch"
echo "     - fashion_mnist_mobilenet_final_1epoch"
echo "     - fashion_mnist_resnet_final_1epoch"
echo ""
echo "=========================================================================="
echo ""
