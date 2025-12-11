"""
Standalone Baseline Model Evaluation for Fashion MNIST

Evaluates all 7 models standalone (baseline, no PCA) for direct comparison.
This creates a dedicated MLflow experiment for easy model comparison.

Usage:
    # Evaluate all models
    uv run python src/evaluate_baseline_models.py --all

    # Evaluate specific model
    uv run python src/evaluate_baseline_models.py --model naive_bayes

    # Quick test with subset
    uv run python src/evaluate_baseline_models.py --all --quick-test --subset-size 1000
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data_loader import FashionMNISTLoader
from src.config import (
    AVAILABLE_MODELS,
    MLFLOW_TRACKING_URI,
    CLASS_NAMES,
    PLOTS_DIR,
)
from src.hyperparameters_fashion_mnist import get_default_params
from src.utils import setup_logger

logger = setup_logger(__name__)


def get_model_family(model_name: str) -> str:
    """Get model family for tagging"""
    if "naive_bayes" in model_name:
        return "naive_bayes"
    elif model_name == "gmm":
        return "gmm"
    elif "logistic" in model_name:
        return "logistic"
    elif model_name == "random_forest":
        return "random_forest"
    else:
        return "unknown"


def get_model_type(model_name: str) -> str:
    """Get model type (generative/discriminative)"""
    generative = ["naive_bayes", "naive_bayes_bernoulli", "naive_bayes_multinomial", "gmm"]
    return "generative" if model_name in generative else "discriminative"


def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, normalize: bool = False):
    """Log confusion matrix as artifact"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title_suffix = " (Normalized)"
        fmt = '.2f'
    else:
        title_suffix = ""
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.title(f"Confusion Matrix - {model_name}{title_suffix}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = f"confusion_matrix_{model_name}{'_normalized' if normalize else ''}.png"
    filepath = PLOTS_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    mlflow.log_artifact(str(filepath))
    plt.close()
    
    logger.info(f"  Saved confusion matrix: {filename}")


def log_per_class_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Log per-class confusion matrices (one vs rest)"""
    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Create binary labels: this class vs all others
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap="Blues",
            xticklabels=['Other', class_name],
            yticklabels=['Other', class_name],
        )
        plt.title(f"Confusion Matrix - {class_name} vs Others\n{model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        filename = f"confusion_matrix_{model_name}_class_{class_name.replace('/', '_')}.png"
        filepath = PLOTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(str(filepath))
        plt.close()


def log_feature_importance(model, model_name: str, n_features: int = 784):
    """Log feature importance for Random Forest"""
    if model_name == "random_forest" and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_n = min(50, len(importances))
        top_indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[top_indices])
        plt.yticks(range(top_n), [f"Feature {i}" for i in top_indices])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filename = f"feature_importance_{model_name}.png"
        filepath = PLOTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(str(filepath))
        plt.close()
        
        logger.info(f"  Saved feature importance: {filename}")
        
        # Log top features as artifact
        top_features_text = "\n".join([
            f"Feature {idx}: {imp:.6f}" 
            for idx, imp in zip(top_indices, importances[top_indices])
        ])
        filename_txt = f"feature_importance_{model_name}.txt"
        filepath_txt = PLOTS_DIR / filename_txt
        filepath_txt.write_text(top_features_text)
        mlflow.log_artifact(str(filepath_txt))


def evaluate_baseline_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_tuned_params: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a single model standalone (baseline, no PCA)
    
    Args:
        model_name: Name of the model
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits
        use_tuned_params: If True, try to load best params from tuning, else use defaults
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"BASELINE EVALUATION: {model_name}")
    logger.info(f"{'=' * 80}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fashion-mnist-baseline-models")
    
    run_name = f"{model_name}_baseline"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Comprehensive tags
        mlflow.set_tags({
            "experiment_type": "baseline",
            "model": model_name,
            "model_family": get_model_family(model_name),
            "model_type": get_model_type(model_name),
            "dataset": "fashion_mnist",
            "has_pca": "false",
            "pca_config": "baseline",
            "has_hyperparameter_tuning": "true" if use_tuned_params else "false",
        })
        
        # Get hyperparameters
        if use_tuned_params:
            # Try to load from tuning experiment
            try:
                from src.final_evaluation import load_best_params
                best_params = load_best_params(model_name)
                logger.info(f"  Using tuned hyperparameters: {best_params}")
            except FileNotFoundError:
                logger.warning(f"  Tuned params not found, using defaults")
                best_params = get_default_params(model_name)
        else:
            best_params = get_default_params(model_name)
        
        # Log dataset parameters
        mlflow.log_params({
            "model_name": model_name,
            "pca_config": "baseline",
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train)),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "random_seed": 42,
        })
        
        # Log model hyperparameters
        mlflow.log_params(best_params)
        
        # Create model instance
        from src.final_evaluation import get_model_instance as get_model_with_params
        model = get_model_with_params(model_name, best_params.copy())
        
        # Combine train+val for final training
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.hstack([y_train, y_val])
        
        # Train model
        logger.info(f"\nTraining {model_name}...")
        start_time = time.time()
        model.fit(X_trainval, y_trainval)
        training_time = time.time() - start_time
        
        logger.info(f"  Training time: {training_time:.2f}s")
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Predictions
        logger.info("\nEvaluating on train, val, and test sets...")
        y_train_pred = model.predict(X_trainval)
        y_test_pred = model.predict(X_test)
        
        # Compute metrics
        def compute_metrics(y_true, y_pred, prefix):
            return {
                f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
                f"{prefix}_precision_macro": precision_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                f"{prefix}_recall_macro": recall_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                f"{prefix}_f1_macro": f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                f"{prefix}_precision_weighted": precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                f"{prefix}_recall_weighted": recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                f"{prefix}_f1_weighted": f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
            }
        
        train_metrics = compute_metrics(y_trainval, y_train_pred, "train")
        test_metrics = compute_metrics(y_test, y_test_pred, "test")
        
        # Inference time
        start_time = time.time()
        _ = model.predict(X_test[:100])
        inference_time = (time.time() - start_time) / 100  # per sample
        test_metrics["inference_time_per_sample"] = inference_time * 1000  # ms
        
        # Log all metrics
        mlflow.log_metrics({**train_metrics, **test_metrics})
        
        # Per-class metrics
        report = classification_report(
            y_test, y_test_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
        
        for class_name in CLASS_NAMES:
            if class_name in report:
                mlflow.log_metric(f"class_{class_name}_precision", report[class_name]["precision"])
                mlflow.log_metric(f"class_{class_name}_recall", report[class_name]["recall"])
                mlflow.log_metric(f"class_{class_name}_f1_score", report[class_name]["f1-score"])
        
        # Model complexity metrics
        if hasattr(model, 'n_features_in_'):
            mlflow.log_param("n_features_in", model.n_features_in_)
        if hasattr(model, 'n_parameters'):
            mlflow.log_metric("n_parameters", model.n_parameters)
        
        # Save model
        mlflow.sklearn.log_model(model, "model")
        
        # Log confusion matrices
        logger.info("\nGenerating visualizations...")
        log_confusion_matrix(y_test, y_test_pred, model_name, normalize=False)
        log_confusion_matrix(y_test, y_test_pred, model_name, normalize=True)
        log_per_class_confusion_matrices(y_test, y_test_pred, model_name)
        
        # Feature importance (for Random Forest)
        if model_name == "random_forest":
            log_feature_importance(model, model_name, n_features=X_train.shape[1])
        
        # Classification report
        report_text = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES, zero_division=0)
        filename = f"classification_report_{model_name}.txt"
        filepath = PLOTS_DIR / filename
        filepath.write_text(report_text)
        mlflow.log_artifact(str(filepath))
        
        logger.info(f"\n✓ {model_name} evaluation complete!")
        logger.info(f"  Test accuracy: {test_metrics['test_accuracy']:.4f}")
        logger.info(f"  Test F1 (macro): {test_metrics['test_f1_macro']:.4f}")
        logger.info(f"  Training time: {training_time:.2f}s")
        logger.info(f"  Inference time: {inference_time*1000:.2f}ms per sample")
        
        return {
            "model_name": model_name,
            "test_accuracy": test_metrics["test_accuracy"],
            "test_f1_macro": test_metrics["test_f1_macro"],
            "training_time": training_time,
            "run_id": run.info.run_id,
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models (standalone, no PCA) for Fashion MNIST"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        help="Specific model to evaluate",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all models",
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
        "--use-defaults",
        action="store_true",
        help="Use default hyperparameters instead of tuned ones",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.model and not args.all:
        logger.error("Must specify --model or --all")
        sys.exit(1)
    
    # Load data
    logger.info("Loading Fashion MNIST...")
    loader = FashionMNISTLoader(flatten=True, normalize=True)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()
    
    if args.quick_test:
        logger.info(f"Using subset of {args.subset_size} samples for quick test")
        X_train = X_train[:args.subset_size]
        y_train = y_train[:args.subset_size]
        X_val = X_val[:args.subset_size // 5]  # Proportional
        y_val = y_val[:args.subset_size // 5]
        X_test = X_test[:args.subset_size // 5]
        y_test = y_test[:args.subset_size // 5]
    
    # Determine models to evaluate
    models_to_evaluate = AVAILABLE_MODELS if args.all else [args.model]
    
    results = []
    for model_name in models_to_evaluate:
        try:
            result = evaluate_baseline_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                use_tuned_params=not args.use_defaults,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}", exc_info=True)
            continue
    
    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'=' * 80}")
    for result in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
        logger.info(
            f"{result['model_name']:25s} | "
            f"Accuracy: {result['test_accuracy']:.4f} | "
            f"F1: {result['test_f1_macro']:.4f} | "
            f"Time: {result['training_time']:.2f}s"
        )
    
    logger.info(f"\n✓ All evaluations complete!")
    logger.info(f"View results: uv run mlflow ui --backend-store-uri file://$(pwd)/results/mlruns")


if __name__ == "__main__":
    main()
