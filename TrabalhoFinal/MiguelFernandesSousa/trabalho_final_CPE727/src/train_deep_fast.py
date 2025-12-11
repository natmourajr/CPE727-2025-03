"""
Fast Deep Learning Training Pipeline

Strategy:
1. Stage 1: Grid search on tiny dataset (500 samples) to find best hyperparameters
2. Final: Train with FULL dataset (60k samples) for 1 EPOCH ONLY using best hyperparameters

This provides:
- ✅ Hyperparameter optimization
- ✅ Training on full dataset
- ✅ Test accuracy on full test set
- ⚡ Fast execution (~28-90 min per model)

Usage:
    # Single model
    uv run python src/train_deep_fast.py --dataset fashion_mnist --model cnn

    # All models
    uv run python src/train_deep_fast.py --dataset fashion_mnist --all
"""
import argparse
import sys
import gc
import time
from pathlib import Path

import numpy as np
import torch
import mlflow
from sklearn.model_selection import StratifiedKFold
from itertools import product

from src.data_loader import FashionMNISTLoader
from src.data_loader_agnews_tokenized import AGNewsTokenizedLoader
from src.models_deep import (
    FashionMNISTCNN, LeNet5Modified, ResNet18Adapted, MobileNetV2Small,
    AGNewsLSTM, TextCNN, BiLSTMSimple, LSTMAttention, DistilBERTClassifier,
    PyTorchClassifier
)
from src.config import MLRUNS_DIR
from src.utils import setup_logger

logger = setup_logger(__name__)

# Import grids from train_deep.py
from src.train_deep import (
    FASHION_MNIST_GRID_STAGE1,
    RESNET_GRID_STAGE1,
    MOBILENET_GRID_STAGE1,
    AG_NEWS_GRID_STAGE1,
    TEXTCNN_GRID_STAGE1,
    BILSTM_GRID_STAGE1,
    DISTILBERT_GRID_STAGE1,
    memory_cleanup,
    prepare_fashion_mnist_data,
    prepare_agnews_data,
    create_model,
    cross_validate,
)


def grid_search_stage1(X, y, dataset, model_type, param_grid, cv_folds=2, experiment_name=None):
    """
    Run Stage 1 grid search (same as original but isolated)
    """
    if experiment_name:
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        mlflow.set_experiment(experiment_name)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    logger.info(f"Grid search: {len(param_combinations)} combinations, {cv_folds}-fold CV")

    results = []
    best_score = 0
    best_params = None

    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))

        # Extract vocab_size if AG News
        if dataset == 'ag_news':
            vocab_size = params.get('vocab_size')
            if isinstance(vocab_size, list):
                params['vocab_size'] = vocab_size[0]
            elif vocab_size is None:
                params['vocab_size'] = 10001

        logger.info(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")

        # Cross-validate
        start_time = time.time()
        mean_score, std_score = cross_validate(X, y, dataset, model_type, cv_folds, **params)
        elapsed_time = time.time() - start_time

        logger.info(f"  CV Accuracy: {mean_score:.4f} (+/- {std_score:.4f}) [{elapsed_time:.1f}s]")

        # Track in MLflow
        if experiment_name:
            with mlflow.start_run(run_name=f"{model_type}_{i+1}"):
                mlflow.log_params(params)
                mlflow.log_metric('cv_accuracy_mean', mean_score)
                mlflow.log_metric('cv_accuracy_std', std_score)
                mlflow.log_metric('cv_time', elapsed_time)

        results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'time': elapsed_time
        })

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            logger.info(f"  *** New best score: {best_score:.4f}")

        memory_cleanup()

    logger.info(f"\n{'='*80}")
    logger.info(f"Stage 1 Complete!")
    logger.info(f"Best CV Accuracy: {best_score:.4f}")
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"{'='*80}")

    return best_params, best_score


def train_final_fast(X_train, y_train, X_test, y_test, dataset, model_type, params, experiment_name=None):
    """
    Train final model with FULL dataset but only 1 EPOCH

    This is much faster than original (20 epochs) but still trains on all data
    """
    logger.info(f"\nTraining final model with best parameters (1 EPOCH ONLY)...")
    logger.info(f"Parameters: {params}")
    logger.info(f"Training set size: {len(X_train)}")

    # Set up MLflow
    if experiment_name:
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        mlflow.set_experiment(experiment_name)

    # Override epochs to 1
    params_final = params.copy()
    params_final['epochs'] = 1  # ONLY 1 EPOCH
    params_final['log_to_mlflow'] = True  # Enable per-epoch logging

    # Create and train model
    start_time = time.time()

    with mlflow.start_run(run_name=f"{model_type}_final_1epoch"):
        classifier = create_model(dataset, model_type, **params_final)
        classifier.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        y_pred = classifier.predict(X_test)
        test_accuracy = (y_pred == y_test).mean()

        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Training time: {train_time:.1f}s")

        # Log to MLflow
        mlflow.log_params(params_final)
        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.log_metric('train_time', train_time)
        mlflow.log_metric('train_samples', len(X_train))
        mlflow.log_metric('epochs_trained', 1)

        # Save model
        try:
            mlflow.pytorch.log_model(classifier.model, "model")
        except Exception as e:
            logger.warning(f"Could not log PyTorch model: {e}")

    return test_accuracy, classifier


def run_fast_pipeline(dataset, model_type):
    """
    Run fast pipeline for a single model:
    1. Stage 1: Grid search on tiny dataset
    2. Final: Train with full dataset, 1 epoch
    """
    logger.info("=" * 80)
    logger.info(f"FAST PIPELINE: {dataset.upper()} - {model_type.upper()}")
    logger.info("=" * 80)
    logger.info("Strategy: Stage 1 (tiny) + Final training (full data, 1 epoch)")
    logger.info("=" * 80)

    # Load data
    if dataset == 'fashion_mnist':
        data = prepare_fashion_mnist_data()

        # Select grid
        if model_type == 'resnet':
            stage1_grid = RESNET_GRID_STAGE1
        elif model_type == 'mobilenet':
            stage1_grid = MOBILENET_GRID_STAGE1
        else:
            stage1_grid = FASHION_MNIST_GRID_STAGE1

    else:  # ag_news
        data = prepare_agnews_data()

        # Select grid
        if model_type == 'textcnn':
            stage1_grid = TEXTCNN_GRID_STAGE1.copy()
        elif model_type in ['bilstm', 'lstm_attn']:
            stage1_grid = BILSTM_GRID_STAGE1.copy()
        elif model_type == 'distilbert':
            stage1_grid = DISTILBERT_GRID_STAGE1.copy()
        else:
            stage1_grid = AG_NEWS_GRID_STAGE1.copy()

        # Add vocab_size
        if model_type != 'distilbert':
            stage1_grid['vocab_size'] = [data['vocab_size']]

    # STAGE 1: Grid search on tiny dataset
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: TINY GRID SEARCH")
    logger.info("=" * 80)

    X_train_tiny, y_train_tiny, X_test_tiny, y_test_tiny = data['tiny']
    experiment_name_stage1 = f"{dataset}_{model_type}_stage1_tiny"

    best_params, best_cv_score = grid_search_stage1(
        X_train_tiny, y_train_tiny,
        dataset, model_type,
        stage1_grid,
        cv_folds=2,
        experiment_name=experiment_name_stage1
    )

    logger.info(f"\n✓ Stage 1 complete - Best CV: {best_cv_score:.4f}")

    # FINAL: Train with full dataset, 1 epoch
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TRAINING: Full Dataset (1 EPOCH ONLY)")
    logger.info("=" * 80)

    X_train_full, y_train_full, X_test_full, y_test_full = data['full']
    experiment_name_final = f"{dataset}_{model_type}_final_1epoch"

    test_accuracy, model = train_final_fast(
        X_train_full, y_train_full, X_test_full, y_test_full,
        dataset, model_type, best_params,
        experiment_name=experiment_name_final
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Model: {model_type.upper()}")
    logger.info(f"Stage 1 Best CV: {best_cv_score:.4f}")
    logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Training samples: {len(X_train_full)}")
    logger.info(f"Epochs: 1")
    logger.info("=" * 80)

    return test_accuracy


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fast deep learning pipeline: Stage 1 + Final (1 epoch)"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['fashion_mnist', 'ag_news'],
        help='Dataset to use'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['cnn', 'lenet', 'resnet', 'mobilenet', 'lstm', 'textcnn', 'bilstm', 'lstm_attn', 'distilbert'],
        help='Model type (required unless --all is used)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Train all models for the dataset'
    )

    args = parser.parse_args()

    # Validation
    if not args.model and not args.all:
        parser.error("Must specify --model or --all")

    # Determine models to train
    if args.all:
        if args.dataset == 'fashion_mnist':
            models = ['cnn', 'lenet', 'mobilenet', 'resnet']
        else:  # ag_news
            models = ['lstm', 'textcnn', 'bilstm', 'lstm_attn']
            # Add distilbert if available
            try:
                from transformers import DistilBertModel
                models.append('distilbert')
            except ImportError:
                logger.warning("transformers not available, skipping distilbert")
    else:
        models = [args.model]

    # Validate models
    if args.dataset == 'fashion_mnist':
        valid_models = ['cnn', 'lenet', 'resnet', 'mobilenet']
    else:
        valid_models = ['lstm', 'textcnn', 'bilstm', 'lstm_attn', 'distilbert']

    for model in models:
        if model not in valid_models:
            parser.error(f"Model {model} not valid for dataset {args.dataset}")

    # Run pipeline for each model
    results = {}
    total_start = time.time()

    for i, model in enumerate(models):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# MODEL {i+1}/{len(models)}: {model.upper()}")
        logger.info(f"{'#'*80}\n")

        try:
            accuracy = run_fast_pipeline(args.dataset, model)
            results[model] = accuracy
        except Exception as e:
            logger.error(f"Failed to train {model}: {e}")
            results[model] = None

        memory_cleanup()

    total_time = time.time() - total_start

    # Final summary
    logger.info("\n\n" + "=" * 80)
    logger.info("ALL MODELS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nDataset: {args.dataset}")
    logger.info(f"Total time: {total_time/60:.1f} minutes\n")
    logger.info("Results:")
    logger.info("-" * 80)
    logger.info(f"{'Model':<15} {'Test Accuracy':<20}")
    logger.info("-" * 80)

    for model, accuracy in results.items():
        if accuracy is not None:
            logger.info(f"{model:<15} {accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            logger.info(f"{model:<15} FAILED")

    logger.info("-" * 80)
    logger.info(f"\nView detailed results in MLflow UI:")
    logger.info(f"  mlflow ui")
    logger.info(f"  Open: http://localhost:5000")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
