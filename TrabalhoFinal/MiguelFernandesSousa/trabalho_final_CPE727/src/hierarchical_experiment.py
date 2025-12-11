"""
Hierarchical Classifier Experiment for Fashion MNIST

Tests 10 different configurations of hierarchical models to find
the best combination of models for each stage.

Configurations test different model combinations:
1. All Logistic Softmax (baseline hierarchical)
2. All Random Forest
3. Recommended mix (from analysis)
4-10. Various other combinations

Each configuration is evaluated on:
- Stage 1 accuracy (group classification)
- Stage 2 specialist accuracies (per group)
- Overall 10-class accuracy
- Comparison with flat baselines

Usage:
    # Run all configurations
    uv run python src/hierarchical_experiment.py

    # Quick test with subset
    uv run python src/hierarchical_experiment.py --quick-test

    # Single configuration
    uv run python src/hierarchical_experiment.py --config 3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data_loader import FashionMNISTLoader
from src.models.hierarchical_classifier import HierarchicalClassifier
from src.models import (
    NaiveBayesGaussian,
    GMMClassifier,
    LogisticRegressionSoftmax,
    LogisticRegressionOvR,
    RandomForest,
)
from src.config import MLFLOW_TRACKING_URI
from src.utils import setup_logger

logger = setup_logger(__name__)


# Experiment configurations
# Format: (stage1_model_name, stage2a_model_name, stage2b_model_name, stage2c_model_name)
EXPERIMENT_CONFIGS = [
    # Config 1: All Logistic Softmax (baseline hierarchical)
    ("logistic_softmax", "logistic_softmax", "logistic_softmax", "logistic_softmax"),
    # Config 2: All Random Forest
    ("random_forest", "random_forest", "random_forest", "random_forest"),
    # Config 3: Recommended mix (from analysis)
    # Stage 1: RF (high accuracy on well-separated groups)
    # Stage 2a (Tops): GMM (handles T-shirt/Shirt confusion with multimodal distributions)
    # Stage 2b (Footwear): Logistic Softmax (structural differences)
    # Stage 2c (Other): Naive Bayes (easy separation)
    ("random_forest", "gmm", "logistic_softmax", "naive_bayes"),
    # Config 4: All GMM
    ("gmm", "gmm", "gmm", "gmm"),
    # Config 5: Logistic stage 1, specialized stage 2
    ("logistic_softmax", "random_forest", "logistic_softmax", "naive_bayes"),
    # Config 6: RF stage 1, all Logistic stage 2
    ("random_forest", "logistic_softmax", "logistic_softmax", "logistic_softmax"),
    # Config 7: Logistic stage 1, RF for hard groups
    ("logistic_softmax", "random_forest", "random_forest", "logistic_softmax"),
    # Config 8: RF stage 1, GMM for all specialists
    ("random_forest", "gmm", "gmm", "gmm"),
    # Config 9: Mix of discriminative models
    ("random_forest", "logistic_softmax", "random_forest", "logistic_ovr"),
    # Config 10: Generative stage 1, discriminative stage 2
    ("gmm", "random_forest", "logistic_softmax", "logistic_softmax"),
]


def get_model_instance(model_name: str, gmm_covariance_type: str = "full"):
    """
    Create model instance by name

    Args:
        model_name: Model identifier
        gmm_covariance_type: Covariance type for GMM models ('full' or 'diag')

    Returns:
        Model instance
    """
    models = {
        "naive_bayes": NaiveBayesGaussian(),
        "gmm": GMMClassifier(n_components=2, covariance_type=gmm_covariance_type, verbose=0, random_state=42, reg_covar=1e-6),
        "logistic_softmax": LogisticRegressionSoftmax(C=1.0, max_iter=2000, verbose=0),
        "logistic_ovr": LogisticRegressionOvR(C=1.0, max_iter=2000, verbose=0),
        "random_forest": RandomForest(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name]


def run_hierarchical_config(
    config_id: int,
    stage1_name: str,
    stage2a_name: str,
    stage2b_name: str,
    stage2c_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gmm_covariance_type: str = "full",
) -> dict:
    """
    Run single hierarchical configuration

    Args:
        config_id: Configuration number (1-10)
        stage1_name, stage2a_name, stage2b_name, stage2c_name: Model names
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits
        gmm_covariance_type: Covariance type for GMM models ('full' or 'diag')

    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"CONFIG {config_id}: Hierarchical Classifier")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Stage 1 (grouping): {stage1_name}")
    logger.info(f"  Stage 2a (Tops): {stage2a_name}")
    logger.info(f"  Stage 2b (Footwear): {stage2b_name}")
    logger.info(f"  Stage 2c (Other): {stage2c_name}")

    # Start MLflow run
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fashion-mnist-hierarchical-comparison")

    # Add covariance type suffix to run name if using non-default
    cov_suffix = f"_cov_{gmm_covariance_type}" if gmm_covariance_type != "full" else ""
    run_name = f"hierarchical_config_{config_id}_{stage1_name}_{stage2a_name}_{stage2b_name}_{stage2c_name}{cov_suffix}"

    with mlflow.start_run(run_name=run_name) as run:
        # Comprehensive tags
        def get_model_family(name: str) -> str:
            if "naive_bayes" in name:
                return "naive_bayes"
            elif name == "gmm":
                return "gmm"
            elif "logistic" in name:
                return "logistic"
            elif name == "random_forest":
                return "random_forest"
            return "unknown"
        
        mlflow.set_tags({
            "experiment_type": "hierarchical",
            "dataset": "fashion_mnist",
            "config_id": str(config_id),
            "stage1_model": stage1_name,
            "stage2a_model": stage2a_name,
            "stage2b_model": stage2b_name,
            "stage2c_model": stage2c_name,
            "gmm_covariance_type": gmm_covariance_type,
            "has_pca": "false",
            "pca_config": "baseline",
        })
        
        mlflow.log_params({
            "config_id": config_id,
            "stage1_model": stage1_name,
            "stage2a_model": stage2a_name,
            "stage2b_model": stage2b_name,
            "stage2c_model": stage2c_name,
            "gmm_covariance_type": gmm_covariance_type,
            "n_features": X_train.shape[1],
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        })

        # Create model instances
        logger.info(f"\nCreating model instances (GMM covariance: {gmm_covariance_type})...")
        stage1_model = get_model_instance(stage1_name, gmm_covariance_type)
        stage2a_model = get_model_instance(stage2a_name, gmm_covariance_type)
        stage2b_model = get_model_instance(stage2b_name, gmm_covariance_type)
        stage2c_model = get_model_instance(stage2c_name, gmm_covariance_type)

        # Create hierarchical classifier
        hierarchical = HierarchicalClassifier(
            stage1_model=stage1_model,
            stage2a_model=stage2a_model,
            stage2b_model=stage2b_model,
            stage2c_model=stage2c_model,
        )

        # Train on train+val
        logger.info("\nTraining on train+val combined...")
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.hstack([y_train, y_val])

        start_time = time.time()
        hierarchical.fit(X_trainval, y_trainval)
        training_time = time.time() - start_time

        logger.info(f"\n  Training time: {training_time:.2f}s")
        mlflow.log_metric("training_time_seconds", training_time)

        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        y_pred = hierarchical.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        test_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        logger.info(f"  Overall test accuracy: {test_acc:.4f}")
        logger.info(f"  Test precision (macro): {test_prec:.4f}")
        logger.info(f"  Test recall (macro): {test_rec:.4f}")
        logger.info(f"  Test F1 (macro): {test_f1:.4f}")

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision_macro", test_prec)
        mlflow.log_metric("test_recall_macro", test_rec)
        mlflow.log_metric("test_f1_macro", test_f1)

        # Get stage metrics
        logger.info("\nStage-wise metrics:")
        stage_metrics = hierarchical.get_stage_metrics(X_test, y_test)

        logger.info(f"  Stage 1 (grouping) accuracy: {stage_metrics['stage1_accuracy']:.4f}")
        mlflow.log_metric("stage1_accuracy", stage_metrics["stage1_accuracy"])

        logger.info(f"  Stage 2 specialist accuracies:")
        for group_name, acc in stage_metrics["stage2_accuracies"].items():
            logger.info(f"    {group_name}: {acc:.4f}")
            mlflow.log_metric(f"stage2_{group_name.lower()}_accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(hierarchical, "hierarchical_model")

        results = {
            "config_id": config_id,
            "stage1_model": stage1_name,
            "stage2a_model": stage2a_name,
            "stage2b_model": stage2b_name,
            "stage2c_model": stage2c_name,
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "stage1_accuracy": stage_metrics["stage1_accuracy"],
            "stage2_accuracies": stage_metrics["stage2_accuracies"],
            "training_time": training_time,
            "run_id": run.info.run_id,
        }

        return results


def run_flat_baseline(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gmm_covariance_type: str = "full",
) -> dict:
    """
    Run flat (non-hierarchical) baseline for comparison

    Args:
        model_name: Name of model to use
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits
        gmm_covariance_type: Covariance type for GMM models ('full' or 'diag')

    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"FLAT BASELINE: {model_name}")
    logger.info(f"{'=' * 80}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fashion-mnist-hierarchical-comparison")

    # Add covariance type suffix to run name if using non-default
    cov_suffix = f"_cov_{gmm_covariance_type}" if gmm_covariance_type != "full" else ""
    run_name = f"flat_baseline_{model_name}{cov_suffix}"

    with mlflow.start_run(run_name=run_name) as run:
        # Comprehensive tags
        def get_model_family(name: str) -> str:
            if "naive_bayes" in name:
                return "naive_bayes"
            elif name == "gmm":
                return "gmm"
            elif "logistic" in name:
                return "logistic"
            elif name == "random_forest":
                return "random_forest"
            return "unknown"
        
        def get_model_type(name: str) -> str:
            generative = ["naive_bayes", "naive_bayes_bernoulli", "naive_bayes_multinomial", "gmm"]
            return "generative" if name in generative else "discriminative"
        
        mlflow.set_tags({
            "experiment_type": "flat_baseline",
            "model": model_name,
            "model_family": get_model_family(model_name),
            "model_type": get_model_type(model_name),
            "dataset": "fashion_mnist",
            "gmm_covariance_type": gmm_covariance_type,
            "has_pca": "false",
            "pca_config": "baseline",
        })
        
        mlflow.log_params({
            "model_name": model_name,
            "gmm_covariance_type": gmm_covariance_type,
            "n_features": X_train.shape[1],
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        })

        # Train
        logger.info(f"  GMM covariance type: {gmm_covariance_type}")
        model = get_model_instance(model_name, gmm_covariance_type)

        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.hstack([y_train, y_val])

        start_time = time.time()
        model.fit(X_trainval, y_trainval)
        training_time = time.time() - start_time

        logger.info(f"  Training time: {training_time:.2f}s")
        mlflow.log_metric("training_time_seconds", training_time)

        # Evaluate
        y_pred = model.predict(X_test)

        from sklearn.metrics import precision_score, recall_score
        
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        test_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        test_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        test_f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        logger.info(f"  Test accuracy: {test_acc:.4f}")
        logger.info(f"  Test F1 (macro): {test_f1:.4f}")

        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "test_precision_macro": test_precision,
            "test_recall_macro": test_recall,
            "test_f1_weighted": test_f1_weighted,
        })
        
        # Save model
        mlflow.sklearn.log_model(model, "model")

        results = {
            "model": model_name,
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "training_time": training_time,
        }

        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hierarchical classifier experiment for Fashion MNIST"
    )

    parser.add_argument(
        "--config",
        type=int,
        choices=range(1, 11),
        help="Run specific configuration (1-10)",
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use small subset for quick testing",
    )

    parser.add_argument(
        "--subset-size",
        type=int,
        default=5000,
        help="Size of subset for quick test (default: 5000)",
    )

    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip flat baseline comparisons",
    )

    parser.add_argument(
        "--gmm-covariance",
        type=str,
        choices=["full", "diag"],
        default="full",
        help="Covariance type for GMM models: 'full' (default) or 'diag'",
    )

    return parser.parse_args()


def main():
    """Main experiment runner"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("HIERARCHICAL CLASSIFIER EXPERIMENT - FASHION MNIST")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading Fashion MNIST dataset...")
    loader = FashionMNISTLoader(flatten=True, normalize=True, normalization_range='-1_1')
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    logger.info(f"  Train set: {X_train.shape}")
    logger.info(f"  Val set: {X_val.shape}")
    logger.info(f"  Test set: {X_test.shape}")

    # Quick test mode
    if args.quick_test:
        logger.info(f"\n⚡ QUICK TEST MODE - Using subset of {args.subset_size} samples")
        X_train = X_train[: args.subset_size]
        y_train = y_train[: args.subset_size]
        X_val = X_val[: args.subset_size // 4]
        y_val = y_val[: args.subset_size // 4]
        X_test = X_test[: args.subset_size // 5]
        y_test = y_test[: args.subset_size // 5]

    # Determine which configs to run
    if args.config:
        configs_to_run = [args.config - 1]  # Convert to 0-indexed
    else:
        configs_to_run = range(len(EXPERIMENT_CONFIGS))

    logger.info(f"\nRunning {len(configs_to_run)} hierarchical configurations")

    # Run hierarchical configs
    all_results = []

    for config_idx in configs_to_run:
        config_id = config_idx + 1  # 1-indexed for display
        stage1, stage2a, stage2b, stage2c = EXPERIMENT_CONFIGS[config_idx]

        try:
            results = run_hierarchical_config(
                config_id=config_id,
                stage1_name=stage1,
                stage2a_name=stage2a,
                stage2b_name=stage2b,
                stage2c_name=stage2c,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                gmm_covariance_type=args.gmm_covariance,
            )
            all_results.append(results)

        except Exception as e:
            logger.error(f"\n❌ Error in config {config_id}: {e}", exc_info=True)
            continue

    # Run flat baselines for comparison
    baseline_results = []

    if not args.skip_baselines:
        logger.info(f"\n{'=' * 80}")
        logger.info("RUNNING FLAT BASELINES FOR COMPARISON")
        logger.info(f"{'=' * 80}")

        baseline_models = ["logistic_softmax", "random_forest", "gmm"]

        for model_name in baseline_models:
            try:
                results = run_flat_baseline(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    gmm_covariance_type=args.gmm_covariance,
                )
                baseline_results.append(results)

            except Exception as e:
                logger.error(f"\n❌ Error in baseline {model_name}: {e}", exc_info=True)
                continue

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    if all_results:
        # Sort by test accuracy
        all_results.sort(key=lambda x: x["test_accuracy"], reverse=True)

        logger.info("\nHierarchical configurations ranked by test accuracy:")
        logger.info("-" * 80)
        logger.info(
            f"{'Rank':<6} {'Config':<8} {'Stage1':<18} {'Stage2a':<18} {'Test Acc':<10} {'Stage1 Acc'}"
        )
        logger.info("-" * 80)

        for rank, result in enumerate(all_results, 1):
            logger.info(
                f"{rank:<6} {result['config_id']:<8} {result['stage1_model']:<18} "
                f"{result['stage2a_model']:<18} {result['test_accuracy']:.4f}     "
                f"{result['stage1_accuracy']:.4f}"
            )

        # Best config
        best = all_results[0]
        logger.info(f"\n🏆 BEST HIERARCHICAL CONFIGURATION:")
        logger.info(f"   Config {best['config_id']}")
        logger.info(f"   Stage 1: {best['stage1_model']}")
        logger.info(f"   Stage 2a (Tops): {best['stage2a_model']}")
        logger.info(f"   Stage 2b (Footwear): {best['stage2b_model']}")
        logger.info(f"   Stage 2c (Other): {best['stage2c_model']}")
        logger.info(f"   Test accuracy: {best['test_accuracy']:.4f}")
        logger.info(f"   Test F1: {best['test_f1_macro']:.4f}")

    if baseline_results:
        logger.info(f"\n{'=' * 80}")
        logger.info("FLAT BASELINES")
        logger.info("-" * 80)

        baseline_results.sort(key=lambda x: x["test_accuracy"], reverse=True)

        for result in baseline_results:
            logger.info(
                f"  {result['model']:<20} Test Acc: {result['test_accuracy']:.4f}  "
                f"F1: {result['test_f1_macro']:.4f}"
            )

        # Comparison
        if all_results:
            best_hierarchical_acc = all_results[0]["test_accuracy"]
            best_flat_acc = baseline_results[0]["test_accuracy"]
            improvement = (best_hierarchical_acc - best_flat_acc) * 100

            logger.info(f"\n{'=' * 80}")
            logger.info("HIERARCHICAL vs FLAT COMPARISON")
            logger.info(f"{'=' * 80}")
            logger.info(f"  Best hierarchical: {best_hierarchical_acc:.4f}")
            logger.info(f"  Best flat: {best_flat_acc:.4f}")
            logger.info(f"  Improvement: {improvement:+.2f} percentage points")

    logger.info("\n" + "=" * 80)
    logger.info("✓ EXPERIMENT COMPLETED")
    logger.info("=" * 80)
    logger.info("\nTo view results in MLflow UI:")
    logger.info("  mlflow ui")
    logger.info("  Open: http://localhost:5000")
    logger.info("\nExperiment: hierarchical-classifier-experiments")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nError: {e}", exc_info=True)
        sys.exit(1)
