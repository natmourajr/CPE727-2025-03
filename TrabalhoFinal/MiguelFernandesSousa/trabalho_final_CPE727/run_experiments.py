#!/usr/bin/env python3
"""
Main script to run all experiments described in the report.

This script executes the complete pipeline:
1. Exploratory Data Analysis (EDA)
2. Baseline models for Fashion MNIST
3. Baseline models for AG_NEWS
4. Deep Learning - CNN for Fashion MNIST
5. Deep Learning - LSTM for AG_NEWS

Usage:
    uv run run_experiments.py [--skip-eda] [--skip-baseline] [--skip-deep] [--dataset DATASET]

Arguments:
    --skip-eda          Skip exploratory data analysis
    --skip-baseline     Skip baseline experiments
    --skip-deep         Skip deep learning experiments
    --dataset DATASET   Run only specific dataset: 'fashion_mnist' or 'ag_news'
    --help             Show this help message
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        print(f"\n✓ Completed in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)


def run_eda(dataset=None):
    """Run exploratory data analysis."""
    print("\n" + "="*80)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)

    scripts = []
    if dataset is None or dataset == 'fashion_mnist':
        scripts.append(('scripts/eda/fashion_mnist_eda.py', 'Fashion MNIST EDA'))
    if dataset is None or dataset == 'ag_news':
        scripts.append(('scripts/eda/ag_news_eda.py', 'AG_NEWS EDA'))

    for script, desc in scripts:
        if not run_command([sys.executable, script], desc):
            return False
    return True


def run_baseline_fashion_mnist():
    """Run baseline experiments for Fashion MNIST."""
    print("\n" + "="*80)
    print("PHASE 2: BASELINE MODELS - FASHION MNIST")
    print("="*80)

    # Phase 1: Hyperparameter tuning
    print("\n--- Phase 2.1: Hyperparameter Optimization ---")
    if not run_command(
        [sys.executable, 'src/hyperparameter_tuning.py'],
        'Grid Search CV for Fashion MNIST baseline models'
    ):
        return False

    # Phase 2: Final evaluation
    print("\n--- Phase 2.2: Final Evaluation on Test Set ---")
    if not run_command(
        [sys.executable, 'src/final_evaluation.py'],
        'Final evaluation of Fashion MNIST baseline models'
    ):
        return False

    return True


def run_baseline_ag_news():
    """Run baseline experiments for AG_NEWS."""
    print("\n" + "="*80)
    print("PHASE 3: BASELINE MODELS - AG_NEWS")
    print("="*80)

    # Phase 1: Hyperparameter tuning
    print("\n--- Phase 3.1: Hyperparameter Optimization (2-stage strategy) ---")
    if not run_command(
        [sys.executable, 'src/hyperparameter_tuning_agnews.py'],
        'Grid Search CV for AG_NEWS baseline models'
    ):
        return False

    # Phase 2: Final evaluation
    print("\n--- Phase 3.2: Final Evaluation on Test Set ---")
    if not run_command(
        [sys.executable, 'src/final_evaluation_agnews.py'],
        'Final evaluation of AG_NEWS baseline models'
    ):
        return False

    return True


def run_deep_learning_cnn():
    """Run CNN experiments for Fashion MNIST."""
    print("\n" + "="*80)
    print("PHASE 4: DEEP LEARNING - CNN (FASHION MNIST)")
    print("="*80)

    # Stage 1: Hyperparameter tuning (2 stages)
    print("\n--- Phase 4.1: Hyperparameter Optimization (2-stage strategy) ---")
    print("This will run Stage 1 (Tiny - 500 samples) and Stage 2 (Small - 5000 samples)")

    if not run_command(
        [sys.executable, 'src/train_deep.py', '--dataset', 'fashion_mnist', '--mode', 'grid_search'],
        'CNN Grid Search (2 stages) for Fashion MNIST'
    ):
        print("\nNote: If grid search was already completed, you can continue to final training.")
        response = input("Continue to final training? (y/n): ")
        if response.lower() != 'y':
            return False

    # Stage 2: Final training
    print("\n--- Phase 4.2: Final Training (Full Dataset, 20 epochs) ---")
    if not run_command(
        [sys.executable, 'scripts/deep_learning/run_lenet_final.py'],
        'CNN final training on full Fashion MNIST'
    ):
        return False

    # Generate confusion matrix
    print("\n--- Phase 4.3: Generate Confusion Matrix ---")
    if not run_command(
        [sys.executable, 'scripts/deep_learning/gen_confusion_fashion_cnn.py'],
        'Generate CNN confusion matrix'
    ):
        print("Warning: Could not generate confusion matrix, but training completed.")

    return True


def run_deep_learning_lstm():
    """Run LSTM experiments for AG_NEWS."""
    print("\n" + "="*80)
    print("PHASE 5: DEEP LEARNING - LSTM (AG_NEWS)")
    print("="*80)

    # Stage 1: Hyperparameter tuning (2 stages)
    print("\n--- Phase 5.1: Hyperparameter Optimization (2-stage strategy) ---")
    print("This will run Stage 1 (Tiny - 1000 samples) and Stage 2 (Small - 10000 samples)")

    if not run_command(
        [sys.executable, 'src/train_deep.py', '--dataset', 'ag_news', '--mode', 'grid_search'],
        'LSTM Grid Search (2 stages) for AG_NEWS'
    ):
        print("\nNote: If grid search was already completed, you can continue to final training.")
        response = input("Continue to final training? (y/n): ")
        if response.lower() != 'y':
            return False

    # Stage 2: Final training
    print("\n--- Phase 5.2: Final Training (Full Dataset, 20 epochs) ---")
    print("Warning: This will take approximately 88.5 minutes")

    if not run_command(
        [sys.executable, 'scripts/deep_learning/train_lstm_final_only.py'],
        'LSTM final training on full AG_NEWS'
    ):
        return False

    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run all experiments for the project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_experiments.py

  # Run only Fashion MNIST experiments
  python run_experiments.py --dataset fashion_mnist

  # Run only AG_NEWS experiments
  python run_experiments.py --dataset ag_news

  # Skip EDA and run only baseline experiments
  python run_experiments.py --skip-eda --skip-deep

  # Run only deep learning experiments
  python run_experiments.py --skip-eda --skip-baseline
        """
    )

    parser.add_argument('--skip-eda', action='store_true',
                       help='Skip exploratory data analysis')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline experiments')
    parser.add_argument('--skip-deep', action='store_true',
                       help='Skip deep learning experiments')
    parser.add_argument('--dataset', choices=['fashion_mnist', 'ag_news'],
                       help='Run only specific dataset experiments')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("TRABALHO FINAL CPE727 - EXPERIMENT PIPELINE")
    print("Comparing Generative vs Discriminative Models")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Skip EDA: {args.skip_eda}")
    print(f"  - Skip Baseline: {args.skip_baseline}")
    print(f"  - Skip Deep Learning: {args.skip_deep}")
    print(f"  - Dataset filter: {args.dataset or 'All'}")
    print()

    start_time = time.time()
    success = True

    # Phase 1: EDA
    if not args.skip_eda:
        if not run_eda(args.dataset):
            print("\n⚠ EDA failed, but continuing with experiments...")

    # Phase 2-3: Baseline experiments
    if not args.skip_baseline:
        if args.dataset is None or args.dataset == 'fashion_mnist':
            if not run_baseline_fashion_mnist():
                success = False
                print("\n✗ Fashion MNIST baseline experiments failed")

        if success and (args.dataset is None or args.dataset == 'ag_news'):
            if not run_baseline_ag_news():
                success = False
                print("\n✗ AG_NEWS baseline experiments failed")

    # Phase 4-5: Deep learning experiments
    if success and not args.skip_deep:
        if args.dataset is None or args.dataset == 'fashion_mnist':
            if not run_deep_learning_cnn():
                success = False
                print("\n✗ CNN experiments failed")

        if success and (args.dataset is None or args.dataset == 'ag_news'):
            if not run_deep_learning_lstm():
                success = False
                print("\n✗ LSTM experiments failed")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EXPERIMENT PIPELINE SUMMARY")
    print("="*80)
    print(f"\nTotal execution time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")

    if success:
        print("\n✓ All experiments completed successfully!")
        print("\nNext steps:")
        print("  1. View results in MLflow UI: mlflow ui")
        print("  2. Analyze results: python scripts/evaluation/analyze_mlflow_results.py")
        print("  3. Check for missing models: python scripts/evaluation/check_missing_models.py")
        print("  4. View confusion matrices in: confusion_matrices_baseline/ and results/plots/")
    else:
        print("\n✗ Some experiments failed. Check the output above for details.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
