#!/bin/bash
# Run missing AG_NEWS models with memory-safe strategy
#
# Strategy:
# 1. Use smaller subset for hyperparameter tuning (Phase 1) to avoid memory explosion
# 2. Use full train+test data for final evaluation (Phase 2) with best hyperparameters
#
# This follows the approach described in GRID_SEARCH_OPTIMIZATION_STRATEGY.md

set -e  # Exit on error

echo "================================================================================"
echo "Running Missing AG_NEWS Models - Memory-Safe Strategy"
echo "================================================================================"
echo ""
echo "Strategy:"
echo "  Phase 1 (Hyperparameter Tuning): Use 30k subset + 2-fold CV"
echo "  Phase 2 (Final Evaluation): Use full train+test data with best params"
echo ""
echo "This prevents memory explosion while still getting accurate final results."
echo "================================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "src/train_agnews.py" ]; then
    echo "❌ Error: Must run from project root (src/train_agnews.py not found)"
    exit 1
fi

# Function to run a model with both phases
run_model() {
    local model=$1
    local subset_size=$2
    local cv_folds=$3
    
    echo ""
    echo "================================================================================"
    echo "Running: $model"
    echo "================================================================================"
    
    # Phase 1: Hyperparameter tuning with subset (memory-safe)
    echo ""
    echo "Phase 1: Hyperparameter Tuning (subset=${subset_size}, cv_folds=${cv_folds})"
    echo "--------------------------------------------------------------------------------"
    uv run python src/train_agnews.py \
        --phase 1 \
        --model "$model" \
        --quick-test \
        --subset-size "$subset_size" \
        --cv-folds "$cv_folds"
    
    if [ $? -ne 0 ]; then
        echo "❌ Phase 1 failed for $model"
        return 1
    fi
    
    echo ""
    echo "✅ Phase 1 completed for $model"
    
    # Phase 2: Final evaluation with full data
    echo ""
    echo "Phase 2: Final Evaluation (full train+test data)"
    echo "--------------------------------------------------------------------------------"
    uv run python src/train_agnews.py \
        --phase 2 \
        --model "$model"
    
    if [ $? -ne 0 ]; then
        echo "❌ Phase 2 failed for $model"
        return 1
    fi
    
    echo ""
    echo "✅ Phase 2 completed for $model"
    echo "✅ $model is fully complete!"
}

# Configuration
SUBSET_SIZE=30000  # 31% of 96k samples (good balance of speed vs accuracy)
CV_FOLDS=2         # 2-fold CV (faster, slightly less stable than 3-fold)

echo "Configuration:"
echo "  Subset size: $SUBSET_SIZE (31% of full dataset)"
echo "  CV folds: $CV_FOLDS"
echo ""

# Run missing models
echo "================================================================================"
echo "Step 1: Run Missing Hyperparameter Tuning Models"
echo "================================================================================"

# Logistic Softmax
run_model "logistic_softmax" "$SUBSET_SIZE" "$CV_FOLDS"

# Logistic OvR
run_model "logistic_ovr" "$SUBSET_SIZE" "$CV_FOLDS"

# Summary
echo ""
echo "================================================================================"
echo "All Missing Models Completed!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Verify results: python3 check_missing_models.py"
echo "  2. View in MLflow UI: mlflow ui"
echo "  3. Check test accuracy in final-evaluation-agnews experiment"
echo ""
echo "✅ Done!"
