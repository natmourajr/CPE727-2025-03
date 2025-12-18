#!/bin/bash

# Quick Training Script - All Fashion MNIST Models (Stage 1 only)
# Estimated time: ~6 minutes total

echo ""
echo "=========================================================================="
echo "QUICK TRAINING - ALL 4 FASHION MNIST MODELS (Stage 1)"
echo "=========================================================================="
echo ""
echo "This will train all 4 models with Stage 1 grid search only."
echo "Stage 1 uses 500 samples to quickly find promising hyperparameters."
echo ""
echo "Estimated time: ~6 minutes total"
echo "  - CNN:       ~35 seconds"
echo "  - LeNet:     ~42 seconds"
echo "  - MobileNet: ~1.2 minutes"
echo "  - ResNet:    ~2.1 minutes"
echo ""
echo "=========================================================================="
echo ""

# Track total time
start_time=$(date +%s)

# Model 1: CNN
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/4] Training CNN (Basic Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
model1_start=$(date +%s)
uv run python src/train_deep.py --dataset fashion_mnist --model cnn --stage 1
model1_end=$(date +%s)
model1_time=$((model1_end - model1_start))
echo ""
echo "✓ CNN completed in ${model1_time}s"
echo ""

# Model 2: LeNet
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/4] Training LeNet-5 (Classic Architecture)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
model2_start=$(date +%s)
uv run python src/train_deep.py --dataset fashion_mnist --model lenet --stage 1
model2_end=$(date +%s)
model2_time=$((model2_end - model2_start))
echo ""
echo "✓ LeNet completed in ${model2_time}s"
echo ""

# Model 3: MobileNet
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/4] Training MobileNetV2 (Efficient Architecture)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
model3_start=$(date +%s)
uv run python src/train_deep.py --dataset fashion_mnist --model mobilenet --stage 1
model3_end=$(date +%s)
model3_time=$((model3_end - model3_start))
echo ""
echo "✓ MobileNet completed in ${model3_time}s"
echo ""

# Model 4: ResNet
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/4] Training ResNet-18 (Deep Residual Network)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
model4_start=$(date +%s)
uv run python src/train_deep.py --dataset fashion_mnist --model resnet --stage 1
model4_end=$(date +%s)
model4_time=$((model4_end - model4_start))
echo ""
echo "✓ ResNet completed in ${model4_time}s"
echo ""

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
total_minutes=$((total_time / 60))
total_seconds=$((total_time % 60))

# Summary
echo ""
echo "=========================================================================="
echo "TRAINING COMPLETE!"
echo "=========================================================================="
echo ""
echo "Individual times:"
echo "  CNN:       ${model1_time}s"
echo "  LeNet:     ${model2_time}s"
echo "  MobileNet: ${model3_time}s"
echo "  ResNet:    ${model4_time}s"
echo ""
echo "Total time: ${total_minutes}m ${total_seconds}s"
echo ""
echo "=========================================================================="
echo "NEXT STEPS"
echo "=========================================================================="
echo ""
echo "1. View results in MLflow UI:"
echo "   mlflow ui"
echo "   Open: http://localhost:5000"
echo ""
echo "2. Compare experiments:"
echo "   - fashion_mnist_cnn_stage1_tiny"
echo "   - fashion_mnist_lenet_stage1_tiny"
echo "   - fashion_mnist_mobilenet_stage1_tiny"
echo "   - fashion_mnist_resnet_stage1_tiny"
echo ""
echo "3. Look for best CV accuracy (cv_accuracy_mean) in each experiment"
echo ""
echo "4. (Optional) Run full pipeline for best model:"
echo "   uv run python src/train_deep.py --dataset fashion_mnist --model <best_model> --full"
echo ""
echo "=========================================================================="
echo ""
