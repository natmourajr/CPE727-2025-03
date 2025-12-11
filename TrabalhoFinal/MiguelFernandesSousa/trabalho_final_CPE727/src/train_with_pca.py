"""
Training Script with PCA Pipeline

Trains models with different PCA configurations for Fashion MNIST.

For each model, trains with:
- baseline (no PCA, 784 features)
- pca_2 (2 components, ~46.8% variance)
- pca_3 (3 components, ~52-55% variance)
- pca_10 (10 components, ~72.0% variance)
- pca_50 (50 components, ~86.3% variance)
- pca_100 (100 components, ~91.2% variance)

Usage:
    # Quick test with one model and subset
    uv run python src/train_with_pca.py --model naive_bayes --quick-test

    # Full training for one model
    uv run python src/train_with_pca.py --model logistic_softmax

    # Train all models with all PCA configs (long!)
    uv run python src/train_with_pca.py --all
"""
import argparse
import sys
from pathlib import Path
import time

import numpy as np
import mlflow

from src.data_loader import FashionMNISTLoader
from src.preprocessing import PCATransformer
from src.config import AVAILABLE_MODELS, PCA_CONFIGS, MLFLOW_TRACKING_URI
from src.hyperparameters_fashion_mnist import get_param_grid, get_default_params
from src.hyperparameter_tuning import get_model_instance
from src.utils import setup_logger

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Fashion MNIST models with PCA experiments"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        help="Specific model to train",
    )

    parser.add_argument(
        "--all", action="store_true", help="Train all models"
    )

    # PCA selection
    parser.add_argument(
        "--pca-config",
        type=str,
        choices=list(PCA_CONFIGS.keys()),
        help="Specific PCA config (default: all configs)",
    )

    # Training mode
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning, use defaults",
    )

    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of CV folds (default: 5)"
    )

    # Quick test
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

    args = parser.parse_args()

    # Validation
    if not args.model and not args.all:
        parser.error("Must specify --model or --all")

    if args.model and args.all:
        parser.error("Cannot specify both --model and --all")

    return args


def train_model_with_pca(
    model_name: str,
    pca_config_name: str,
    n_components: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    skip_tuning: bool = False,
    cv_folds: int = 5,
):
    """
    Train a single model with a specific PCA configuration

    Args:
        model_name: Name of the model
        pca_config_name: Name of PCA config (e.g., 'pca_10', 'baseline')
        n_components: Number of PCA components (None for baseline)
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits
        skip_tuning: If True, use default params instead of grid search
        cv_folds: Number of CV folds

    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING: {model_name} with {pca_config_name}")
    logger.info(f"{'=' * 80}")

    # Start MLflow run
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fashion-mnist-pca-ablation")

    run_name = f"{model_name}_{pca_config_name}"

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
            "experiment_type": "pca_ablation",
            "model": model_name,
            "model_family": get_model_family(model_name),
            "model_type": get_model_type(model_name),
            "dataset": "fashion_mnist",
            "pca_config": pca_config_name,
            "has_pca": "false" if n_components is None else "true",
            "has_hyperparameter_tuning": "false" if skip_tuning else "true",
        })
        
        mlflow.log_param("pca_components", n_components if n_components else "baseline")

        # Apply PCA if needed
        if n_components is not None:
            logger.info(f"\nApplying PCA with {n_components} components...")
            pca_transformer = PCATransformer(n_components=n_components, random_state=42)

            # Fit on train, transform all sets
            X_train_pca, X_val_pca, X_test_pca = pca_transformer.fit_transform(
                X_train, X_val, X_test
            )

            # Log PCA variance info
            variance_info = pca_transformer.get_variance_info()
            mlflow.log_metric(
                "pca_variance_explained", variance_info["total_variance"]
            )
            mlflow.log_param("pca_variance_threshold", f"{variance_info['total_variance']:.2f}")
            mlflow.log_metric("pca_compression_ratio", X_train.shape[1] / n_components)
            
            logger.info(
                f"  PCA variance explained: {variance_info['total_variance']:.2f}%"
            )
            logger.info(
                f"  Compression ratio: {X_train.shape[1] / n_components:.2f}x"
            )

            # Use PCA-transformed data
            X_train_use = X_train_pca
            X_val_use = X_val_pca
            X_test_use = X_test_pca
        else:
            logger.info("\nUsing baseline (no PCA)")
            X_train_use = X_train
            X_val_use = X_val
            X_test_use = X_test

        logger.info(f"  Training shape: {X_train_use.shape}")
        logger.info(f"  Validation shape: {X_val_use.shape}")
        logger.info(f"  Test shape: {X_test_use.shape}")

        # Get hyperparameters
        if skip_tuning:
            logger.info("\nUsing default hyperparameters (no tuning)...")
            best_params = get_default_params(model_name)
            best_score = None
        else:
            logger.info(f"\nHyperparameter tuning with {cv_folds}-fold CV...")
            param_grid = get_param_grid(model_name)
            model = get_model_instance(model_name)

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
            )

            start_time = time.time()
            grid_search.fit(X_train_use, y_train)
            tuning_time = time.time() - start_time

            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            logger.info(f"  Best CV score: {best_score:.4f}")
            logger.info(f"  Best params: {best_params}")
            logger.info(f"  Tuning time: {tuning_time:.2f}s")

            mlflow.log_metric("best_cv_score", best_score)
            mlflow.log_metric("tuning_time_seconds", tuning_time)

        mlflow.log_params(best_params)

        # Train final model on train+val
        logger.info("\nTraining final model on train+val...")
        X_trainval = np.vstack([X_train_use, X_val_use])
        y_trainval = np.hstack([y_train, y_val])

        # Create model with best params
        from src.final_evaluation import get_model_instance as get_model_with_params

        final_model = get_model_with_params(model_name, best_params.copy())

        start_time = time.time()
        final_model.fit(X_trainval, y_trainval)
        training_time = time.time() - start_time

        logger.info(f"  Training time: {training_time:.2f}s")
        mlflow.log_metric("training_time_seconds", training_time)

        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        y_pred = final_model.predict(X_test_use)

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        test_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        logger.info(f"  Test Precision (macro): {test_prec:.4f}")
        logger.info(f"  Test Recall (macro): {test_rec:.4f}")
        logger.info(f"  Test F1 (macro): {test_f1:.4f}")

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision_macro", test_prec)
        mlflow.log_metric("test_recall_macro", test_rec)
        mlflow.log_metric("test_f1_macro", test_f1)

        # Log model
        mlflow.sklearn.log_model(final_model, "model")

        results = {
            "model": model_name,
            "pca_config": pca_config_name,
            "n_components": n_components,
            "best_params": best_params,
            "best_cv_score": best_score,
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "run_id": run.info.run_id,
        }

        return results


def main():
    """Main training pipeline"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("FASHION MNIST TRAINING WITH PCA EXPERIMENTS")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading Fashion MNIST dataset...")

    # Check if training MultinomialNB (needs [0,1] normalization)
    model_list = AVAILABLE_MODELS if args.all else [args.model]
    needs_0_1_norm = "naive_bayes_multinomial" in model_list

    if needs_0_1_norm and len(model_list) > 1:
        logger.warning(
            "\n⚠️  WARNING: MultinomialNB requires [0,1] normalization, "
            "but other models use [-1,1]."
        )
        logger.warning(
            "    Please train MultinomialNB separately with --model naive_bayes_multinomial"
        )
        logger.warning("    or train only MultinomialNB in this run.\n")
        sys.exit(1)

    normalization_range = '0_1' if needs_0_1_norm else '-1_1'
    logger.info(f"  Using normalization range: {normalization_range}")

    loader = FashionMNISTLoader(flatten=True, normalize=True, normalization_range=normalization_range)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    logger.info(f"  Train set: {X_train.shape}")
    logger.info(f"  Val set: {X_val.shape}")
    logger.info(f"  Test set: {X_test.shape}")

    # Quick test mode
    if args.quick_test:
        logger.info(
            f"\n⚡ QUICK TEST MODE - Using subset of {args.subset_size} samples"
        )
        X_train = X_train[: args.subset_size]
        y_train = y_train[: args.subset_size]
        X_val = X_val[: args.subset_size // 4]
        y_val = y_val[: args.subset_size // 4]
        X_test = X_test[: args.subset_size // 5]
        y_test = y_test[: args.subset_size // 5]

    # Determine models to train
    if args.all:
        models_to_train = AVAILABLE_MODELS
    else:
        models_to_train = [args.model]

    # Determine PCA configs to use
    if args.pca_config:
        pca_configs_to_use = {args.pca_config: PCA_CONFIGS[args.pca_config]}
    else:
        pca_configs_to_use = PCA_CONFIGS

    logger.info(f"\nModels to train: {models_to_train}")
    logger.info(f"PCA configs: {list(pca_configs_to_use.keys())}")
    logger.info(
        f"Total experiments: {len(models_to_train)} models × {len(pca_configs_to_use)} configs = {len(models_to_train) * len(pca_configs_to_use)}"
    )

    # Train all combinations
    all_results = []

    for model_name in models_to_train:
        for pca_config_name, n_components in pca_configs_to_use.items():
            try:
                results = train_model_with_pca(
                    model_name=model_name,
                    pca_config_name=pca_config_name,
                    n_components=n_components,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    skip_tuning=args.skip_tuning,
                    cv_folds=args.cv_folds,
                )
                all_results.append(results)

            except Exception as e:
                logger.error(
                    f"\n❌ Error training {model_name} with {pca_config_name}: {e}",
                    exc_info=True,
                )
                continue

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY OF ALL EXPERIMENTS")
    logger.info("=" * 80)

    if all_results:
        # Sort by test accuracy
        all_results.sort(key=lambda x: x["test_accuracy"], reverse=True)

        logger.info("\nTop 10 configurations by test accuracy:")
        logger.info("-" * 80)
        logger.info(
            f"{'Rank':<6} {'Model':<25} {'PCA':<12} {'Components':<12} {'Test Acc':<10} {'Test F1'}"
        )
        logger.info("-" * 80)

        for rank, result in enumerate(all_results[:10], 1):
            model = result["model"]
            pca = result["pca_config"]
            n_comp = result["n_components"] if result["n_components"] else "baseline"
            acc = result["test_accuracy"]
            f1 = result["test_f1_macro"]

            logger.info(
                f"{rank:<6} {model:<25} {pca:<12} {str(n_comp):<12} {acc:.4f}     {f1:.4f}"
            )

        # Best per model
        logger.info("\n" + "=" * 80)
        logger.info("BEST PCA CONFIG PER MODEL")
        logger.info("=" * 80)

        for model_name in models_to_train:
            model_results = [r for r in all_results if r["model"] == model_name]
            if model_results:
                best = max(model_results, key=lambda x: x["test_accuracy"])
                logger.info(f"\n{model_name}:")
                logger.info(f"  Best config: {best['pca_config']}")
                logger.info(f"  Components: {best['n_components']}")
                logger.info(f"  Test accuracy: {best['test_accuracy']:.4f}")
                logger.info(f"  Test F1: {best['test_f1_macro']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL EXPERIMENTS COMPLETED")
    logger.info("=" * 80)
    logger.info("\nTo view results in MLflow UI:")
    logger.info("  mlflow ui")
    logger.info("  Open: http://localhost:5000")
    logger.info("\nExperiment: fashion-mnist-pca-experiments")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nError: {e}", exc_info=True)
        sys.exit(1)
