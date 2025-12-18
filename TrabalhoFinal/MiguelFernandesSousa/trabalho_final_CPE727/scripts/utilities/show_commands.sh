#!/bin/bash
# Summary of what needs to be run for AG_NEWS models
#
# This script doesn't actually run anything - it just prints the commands
# you should run based on current status.

echo "================================================================================"
echo "AG_NEWS Models - Status and Commands to Run"
echo "================================================================================"
echo ""

# Check status
echo "Running status check..."
python3 check_missing_models.py > /tmp/ag_news_status.txt 2>&1

if [ $? -eq 0 ]; then
    cat /tmp/ag_news_status.txt
else
    echo "❌ Could not run check_missing_models.py"
    echo "Make sure you're in the project root directory"
    exit 1
fi

echo ""
echo "================================================================================"
echo "RECOMMENDED COMMANDS"
echo "================================================================================"
echo ""

echo "Option 1: Run everything automatically (RECOMMENDED)"
echo "------------------------------------------------------"
echo "This will run missing models and then final evaluation for all models."
echo ""
echo "  ./run_missing_models.sh       # Run logistic_softmax and logistic_ovr"
echo "  ./run_phase2_evaluations.sh   # Run final evaluation for all 5 completed models"
echo ""

echo "Option 2: Run individual models manually"
echo "-----------------------------------------"
echo "Run missing models (Phase 1 + Phase 2):"
echo ""
echo "  # Logistic Softmax (subset approach for safety)"
echo "  uv run python src/train_agnews.py --phase 1 --model logistic_softmax \\"
echo "    --quick-test --subset-size 30000 --cv-folds 2"
echo "  uv run python src/train_agnews.py --phase 2 --model logistic_softmax"
echo ""
echo "  # Logistic OvR (subset approach for safety)"
echo "  uv run python src/train_agnews.py --phase 1 --model logistic_ovr \\"
echo "    --quick-test --subset-size 30000 --cv-folds 2"
echo "  uv run python src/train_agnews.py --phase 2 --model logistic_ovr"
echo ""

echo "Run Phase 2 (Final Evaluation) for completed models:"
echo ""
echo "  uv run python src/train_agnews.py --phase 2 --model naive_bayes"
echo "  uv run python src/train_agnews.py --phase 2 --model naive_bayes_bernoulli"
echo "  uv run python src/train_agnews.py --phase 2 --model naive_bayes_multinomial"
echo "  uv run python src/train_agnews.py --phase 2 --model gmm"
echo "  uv run python src/train_agnews.py --phase 2 --model random_forest"
echo ""

echo "Option 3: Full dataset approach (if you have 64GB+ RAM)"
echo "---------------------------------------------------------"
echo "Only use this if you're confident your machine can handle it."
echo ""
echo "  uv run python src/train_agnews.py --full --model logistic_softmax --cv-folds 3"
echo "  uv run python src/train_agnews.py --full --model logistic_ovr --cv-folds 3"
echo ""

echo "================================================================================"
echo "MEMORY CONSIDERATIONS"
echo "================================================================================"
echo ""
echo "Subset approach (RECOMMENDED for safety):"
echo "  - Phase 1: Uses 30k samples (31% of dataset) with 2-fold CV"
echo "  - Expected memory: 12-20 GB"
echo "  - Time: ~20-30 min per model"
echo ""
echo "Full dataset approach:"
echo "  - Phase 1: Uses full 96k samples with 3-fold CV"
echo "  - Expected memory: 16-32 GB (logistic models are memory-efficient)"
echo "  - Time: ~15-30 min per model"
echo ""
echo "Random Forest is SAFE now (n_jobs=1 fix already in code)"
echo "  - Old behavior: 96 GB (nested parallelism)"
echo "  - New behavior: 12-20 GB (fixed in src/hyperparameter_tuning_agnews.py:67)"
echo ""

echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"
echo ""
echo "After running, verify completion with:"
echo "  python3 check_missing_models.py"
echo ""
echo "View results in MLflow:"
echo "  mlflow ui"
echo "  Open: http://localhost:5000"
echo ""

echo "================================================================================"
echo "TIME ESTIMATES"
echo "================================================================================"
echo ""
echo "Phase 1 (Hyperparameter Tuning):"
echo "  - Logistic models: ~20-30 min each (with subset)"
echo "  - Total for 2 models: ~40-60 min"
echo ""
echo "Phase 2 (Final Evaluation):"
echo "  - Naive Bayes models: ~5 min each"
echo "  - GMM: ~10 min"
echo "  - Random Forest: ~15 min"
echo "  - Logistic models: ~10 min each"
echo "  - Total for 7 models: ~60-90 min"
echo ""
echo "GRAND TOTAL: ~2-2.5 hours"
echo ""
