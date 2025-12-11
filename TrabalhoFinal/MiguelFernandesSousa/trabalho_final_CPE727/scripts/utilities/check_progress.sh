#!/bin/bash
# Quick script to check AG_NEWS experiment progress

echo "======================================================================"
echo "AG_NEWS EXPERIMENT PROGRESS CHECK"
echo "======================================================================"
echo ""

# Check if Random Forest is still running
echo "1. CURRENTLY RUNNING PROCESSES:"
echo "----------------------------------------------------------------------"
RF_PROC=$(ps aux | grep "train_agnews.*random_forest" | grep -v grep)
if [ -n "$RF_PROC" ]; then
    echo "✅ Random Forest is RUNNING"
    echo "$RF_PROC" | awk '{print "   PID:", $2, "| Runtime:", $10, "| CPU:", $3"%", "| Memory:", $4"%"}'

    # Extract start time
    START_TIME=$(echo "$RF_PROC" | awk '{print $9}')
    echo "   Started at: $START_TIME"
else
    echo "❌ Random Forest is NOT running"
fi
echo ""

# Check memory usage
echo "2. MEMORY USAGE (Top Python Processes):"
echo "----------------------------------------------------------------------"
ps aux | grep python | grep -v grep | sort -k4 -r | head -3 | awk '{printf "   PID: %-8s | Memory: %5s%% | CPU: %5s%% | %s\n", $2, $4, $3, $11}'
echo ""

# Check MLflow experiments
echo "3. MLFLOW EXPERIMENT STATUS:"
echo "----------------------------------------------------------------------"
cd /Users/miguel/Developer/msc/disciplinas/trabalho_final_CPE775
uv run python -c "
import mlflow
import os
from datetime import datetime

os.environ['MLFLOW_TRACKING_URI'] = 'file:///Users/miguel/Developer/msc/disciplinas/trabalho_final_CPE775/results/mlruns'
mlflow.set_tracking_uri('file:///Users/miguel/Developer/msc/disciplinas/trabalho_final_CPE775/results/mlruns')

# Check hyperparameter-tuning-agnews
exp = mlflow.get_experiment_by_name('hyperparameter-tuning-agnews')
if exp:
    runs = mlflow.search_runs(exp.experiment_id, order_by=['start_time DESC'], max_results=10)

    print('Recent runs (last 10):')
    for idx, row in runs.iterrows():
        model = row.get('tags.model', 'unknown')
        status = row.get('status', 'unknown')
        start_time = row.get('start_time', None)
        end_time = row.get('end_time', None)

        if start_time:
            start_str = start_time.strftime('%H:%M:%S')
        else:
            start_str = 'N/A'

        if end_time:
            end_str = end_time.strftime('%H:%M:%S')
            duration = (end_time - start_time).total_seconds() / 60
            duration_str = f'{duration:.1f}m'
        else:
            end_str = 'IN PROGRESS'
            duration_str = '---'

        print(f'   {model:30s} | {status:10s} | {start_str} -> {end_str} | {duration_str}')
" 2>/dev/null

echo ""

# Check best_params directory
echo "4. GENERATED ARTIFACTS:"
echo "----------------------------------------------------------------------"
echo "CV Distribution Plots:"
ls -lh results/best_params_agnews/*_cv_distribution.png 2>/dev/null | awk '{print "   ✅", $9, "(" $5 ")"}'

echo ""
echo "Best Params JSON:"
EXPECTED_MODELS=("naive_bayes" "naive_bayes_bernoulli" "naive_bayes_multinomial" "gmm" "logistic_softmax" "logistic_ovr" "random_forest")
for model in "${EXPECTED_MODELS[@]}"; do
    if [ -f "results/best_params_agnews/${model}_best_params.json" ]; then
        echo "   ✅ ${model}_best_params.json"
    else
        echo "   ❌ ${model}_best_params.json (missing)"
    fi
done

echo ""
echo "======================================================================"
echo "NEXT STEPS:"
echo "======================================================================"
echo ""

if [ -n "$RF_PROC" ]; then
    echo "⏳ Random Forest is still running. While you wait, you can:"
    echo ""
    echo "   Run these FAST experiments in parallel (5-20 min each):"
    echo ""
    echo "   # Terminal 2:"
    echo "   uv run python src/train_agnews.py --phase 1 --model naive_bayes --cv-folds 3"
    echo ""
    echo "   # Terminal 3:"
    echo "   uv run python src/train_agnews.py --phase 1 --model gmm --cv-folds 3"
    echo ""
    echo "   # Terminal 4:"
    echo "   uv run python src/train_agnews.py --phase 1 --model logistic_softmax --cv-folds 3"
    echo ""
    echo "   # Terminal 5:"
    echo "   uv run python src/train_agnews.py --phase 1 --model logistic_ovr --cv-folds 3"
else
    echo "✅ Random Forest completed! Now run:"
    echo ""
    echo "   # Final evaluations for new models:"
    echo "   uv run python src/train_agnews.py --phase 2 --model naive_bayes_bernoulli"
    echo "   uv run python src/train_agnews.py --phase 2 --model random_forest"
    echo ""
    echo "   # Re-run old models with new logging (if needed):"
    echo "   uv run python src/train_agnews.py --phase 1 --model naive_bayes --cv-folds 3"
    echo "   uv run python src/train_agnews.py --phase 1 --model gmm --cv-folds 3"
    echo "   uv run python src/train_agnews.py --phase 1 --model logistic_softmax --cv-folds 3"
    echo "   uv run python src/train_agnews.py --phase 1 --model logistic_ovr --cv-folds 3"
fi

echo ""
echo "======================================================================"
