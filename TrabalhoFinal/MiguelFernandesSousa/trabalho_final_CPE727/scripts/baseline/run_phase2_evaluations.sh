#!/bin/bash
# Run final evaluation (Phase 2) for models that already have hyperparameter tuning
#
# These models have completed Phase 1 (hyperparameter tuning) but are missing
# Phase 2 (final evaluation on test set):
#   - naive_bayes
#   - naive_bayes_bernoulli
#   - naive_bayes_multinomial
#   - gmm
#   - random_forest

set -e  # Exit on error

echo "================================================================================"
echo "Running Final Evaluation (Phase 2) for Completed Models"
echo "================================================================================"
echo ""
echo "These models already have best hyperparameters from Phase 1."
echo "Now running final evaluation on test set with full train data."
echo "================================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "src/train_agnews.py" ]; then
    echo "❌ Error: Must run from project root (src/train_agnews.py not found)"
    exit 1
fi

# Function to run Phase 2 for a model
run_phase2() {
    local model=$1
    
    echo ""
    echo "================================================================================"
    echo "Final Evaluation: $model"
    echo "================================================================================"
    
    uv run python src/train_agnews.py \
        --phase 2 \
        --model "$model"
    
    if [ $? -ne 0 ]; then
        echo "❌ Phase 2 failed for $model"
        return 1
    fi
    
    echo "✅ Phase 2 completed for $model"
}

# Models that need Phase 2
MODELS=(
    "naive_bayes"
    "naive_bayes_bernoulli"
    "naive_bayes_multinomial"
    "gmm"
    "random_forest"
)

echo "Models to evaluate:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Run Phase 2 for each model
for model in "${MODELS[@]}"; do
    run_phase2 "$model"
done

# Summary
echo ""
echo "================================================================================"
echo "All Final Evaluations Completed!"
echo "================================================================================"
echo ""
echo "Results saved in MLflow experiment: final-evaluation-agnews"
echo ""
echo "View results:"
echo "  mlflow ui"
echo "  Open: http://localhost:5000"
echo ""
echo "✅ Done!"
