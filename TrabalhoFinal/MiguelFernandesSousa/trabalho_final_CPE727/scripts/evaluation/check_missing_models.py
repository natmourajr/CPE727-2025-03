#!/usr/bin/env python3
"""
Check which AG_NEWS models are missing from MLflow experiments
"""
import os
from pathlib import Path

# Expected models from DIAGNOSTICO_AG_NEWS.md
EXPECTED_MODELS = [
    "naive_bayes",              # Gaussian Naive Bayes
    "naive_bayes_bernoulli",    # Bernoulli Naive Bayes - MISSING
    "naive_bayes_multinomial",  # Multinomial Naive Bayes - MISSING
    "gmm",                      # GMM
    "logistic_softmax",         # Logistic Softmax - MISSING
    "logistic_ovr",             # Logistic OvR - MISSING
    "random_forest",            # Random Forest - MISSING
]

def check_models():
    """Check which models have run in MLflow"""
    mlruns_dir = Path("results/mlruns")
    
    # Find hyperparameter-tuning-agnews experiment
    hyperparameter_exp = None
    final_eval_exp = None
    
    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ['.trash', 'models']:
            meta_file = exp_dir / "meta.yaml"
            if meta_file.exists():
                content = meta_file.read_text()
                if "hyperparameter-tuning-agnews" in content:
                    hyperparameter_exp = exp_dir
                elif "final-evaluation-agnews" in content:
                    final_eval_exp = exp_dir
    
    if not hyperparameter_exp:
        print("❌ No hyperparameter-tuning-agnews experiment found!")
        return
    
    print(f"✅ Found hyperparameter-tuning-agnews: {hyperparameter_exp.name}")
    
    # Check which models have run
    models_with_runs = set()
    
    for run_dir in hyperparameter_exp.iterdir():
        if run_dir.is_dir():
            model_tag_file = run_dir / "tags" / "model"
            if model_tag_file.exists():
                model = model_tag_file.read_text().strip()
                models_with_runs.add(model)
    
    print(f"\n{'='*80}")
    print("MODELS STATUS (AG_NEWS)")
    print(f"{'='*80}\n")
    
    # Check each expected model
    missing_models = []
    completed_models = []
    
    for model in EXPECTED_MODELS:
        if model in models_with_runs:
            print(f"✅ {model:<30} - Has hyperparameter tuning runs")
            completed_models.append(model)
        else:
            print(f"❌ {model:<30} - MISSING hyperparameter tuning runs")
            missing_models.append(model)
    
    # Check final evaluation
    if final_eval_exp:
        print(f"\n✅ Found final-evaluation-agnews: {final_eval_exp.name}")
        
        final_eval_models = set()
        for run_dir in final_eval_exp.iterdir():
            if run_dir.is_dir():
                model_tag_file = run_dir / "tags" / "model"
                if model_tag_file.exists():
                    model = model_tag_file.read_text().strip()
                    final_eval_models.add(model)
        
        print(f"\n{'='*80}")
        print("FINAL EVALUATION STATUS")
        print(f"{'='*80}\n")
        
        for model in completed_models:
            if model in final_eval_models:
                print(f"✅ {model:<30} - Has final evaluation")
            else:
                print(f"⚠️  {model:<30} - Missing final evaluation")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total expected models: {len(EXPECTED_MODELS)}")
    print(f"Completed (hyperparameter tuning): {len(completed_models)}")
    print(f"Missing (hyperparameter tuning): {len(missing_models)}")
    
    if missing_models:
        print(f"\n{'='*80}")
        print("MISSING MODELS TO RUN")
        print(f"{'='*80}\n")
        
        for model in missing_models:
            print(f"❌ {model}")
            
            # Check memory considerations
            if model == "random_forest":
                print(f"   ⚠️  Memory: Uses n_jobs=1 to avoid nested parallelism")
                print(f"   ⚠️  Strategy: Run with --quick-test first (5k samples)")
                print(f"   💡 Command: uv run python src/train_agnews.py --phase 1 --model {model} --quick-test --subset-size 5000 --cv-folds 2")
            elif model in ["logistic_softmax", "logistic_ovr"]:
                print(f"   ✅ Memory: Should be fine with full dataset")
                print(f"   💡 Command: uv run python src/train_agnews.py --full --model {model} --cv-folds 3")
            else:
                print(f"   ✅ Memory: Should be fine with full dataset")
                print(f"   💡 Command: uv run python src/train_agnews.py --full --model {model} --cv-folds 3")
    else:
        print("\n🎉 All models have been run!")
    
    # Check best_params files
    print(f"\n{'='*80}")
    print("BEST PARAMETERS FILES")
    print(f"{'='*80}\n")
    
    best_params_dir = Path("results/best_params_agnews")
    if best_params_dir.exists():
        for model in EXPECTED_MODELS:
            best_params_file = best_params_dir / f"{model}_best_params.json"
            if best_params_file.exists():
                print(f"✅ {model:<30} - best_params.json exists")
            else:
                print(f"❌ {model:<30} - best_params.json MISSING")
    else:
        print("❌ results/best_params_agnews/ directory not found!")

if __name__ == "__main__":
    check_models()
