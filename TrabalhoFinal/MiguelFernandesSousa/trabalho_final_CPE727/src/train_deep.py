"""
Training script for Deep Learning Models with Memory-Efficient Hyperparameter Tuning

This script implements a two-stage hyperparameter search:
1. Stage 1 (TINY): Quick search on tiny subset to identify important hyperparameters
2. Stage 2 (SMALL): Refined search on small subset with narrowed hyperparameter space

Usage:
    # Fashion MNIST CNN
    uv run python src/train_deep.py --dataset fashion_mnist --model cnn --stage 1
    uv run python src/train_deep.py --dataset fashion_mnist --model cnn --stage 2
    uv run python src/train_deep.py --dataset fashion_mnist --model cnn --full

    # AG News LSTM
    uv run python src/train_deep.py --dataset ag_news --model lstm --stage 1
    uv run python src/train_deep.py --dataset ag_news --model lstm --stage 2
    uv run python src/train_deep.py --dataset ag_news --model lstm --full
"""
import argparse
import sys
import gc
from pathlib import Path
from typing import Dict, Any, Tuple
import time

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


# Hyperparameter grids for Fashion MNIST models
# Note: These grids work for CNN, LeNet, ResNet, and MobileNet
# MobileNet automatically uses lower dropout (0.2) if not specified

FASHION_MNIST_GRID_STAGE1 = {
    'learning_rate': [0.001, 0.01],
    'dropout': [0.3, 0.5],
    'batch_size': [32, 64],
    'epochs': [3],  # Very few epochs for quick test
    # Architecture search (kept compact for stage 1)
    'conv_channels': [(32, 64), (32, 64, 128)],
    'kernel_size': [3],
    'fc_units': [128, 256],
    'use_batchnorm': [False, True],
    'padding': ['same'],
}

FASHION_MNIST_GRID_STAGE2 = {
    'learning_rate': [0.0005, 0.001, 0.002],
    'dropout': [0.3, 0.4, 0.5],
    'batch_size': [32, 64],
    'epochs': [10],
    # Refine with limited architecture options (to keep runtime reasonable)
    'conv_channels': [(32, 64), (32, 64, 128)],
    'kernel_size': [3],
    'fc_units': [128],
    'use_batchnorm': [True],
    'padding': ['same'],
}

# For ResNet (larger model, may need different learning rates)
RESNET_GRID_STAGE1 = {
    'learning_rate': [0.0001, 0.001],
    'dropout': [0.3, 0.5],
    'batch_size': [32, 64],
    'epochs': [3],
}

RESNET_GRID_STAGE2 = {
    'learning_rate': [0.0001, 0.0005, 0.001],
    'dropout': [0.3, 0.4, 0.5],
    'batch_size': [32, 64],
    'epochs': [10],
}

# For MobileNet (efficient model, uses lower dropout)
MOBILENET_GRID_STAGE1 = {
    'learning_rate': [0.001, 0.01],
    'dropout': [0.1, 0.2],
    'batch_size': [32, 64],
    'epochs': [3],
}

MOBILENET_GRID_STAGE2 = {
    'learning_rate': [0.0005, 0.001, 0.002],
    'dropout': [0.1, 0.2, 0.3],
    'batch_size': [32, 64],
    'epochs': [10],
}

# AG News hyperparameter grids (for basic LSTM)
AG_NEWS_GRID_STAGE1 = {
    'learning_rate': [0.001, 0.01],
    'embedding_dim': [50, 100],
    'hidden_dim': [64, 128],
    'dropout': [0.3, 0.5],
    'bidirectional': [False, True],
    'batch_size': [32, 64],
    'epochs': [3],
}

AG_NEWS_GRID_STAGE2 = {
    # Stage 1 best: 0.01, refine around that value (test slightly lower and same)
    'learning_rate': [0.005, 0.01],
    'embedding_dim': [100],  # Stage 1 best
    # Stage 1 best: 64, test both 64 and 128 for refinement
    'hidden_dim': [64, 128],
    # Stage 1 best: 0.3, refine around that (test 0.2, 0.3, 0.4)
    'dropout': [0.2, 0.3, 0.4],
    'bidirectional': [True],  # Stage 1 best
    'batch_size': [32],  # Stage 1 best: 32, focus on that
    'epochs': [5],  # Reduced from 10 to speed up (2x faster)
}

# TextCNN grids (uses 300d embeddings)
TEXTCNN_GRID_STAGE1 = {
    'learning_rate': [0.001, 0.01],
    'embedding_dim': [300],
    'num_filters': [100, 128],
    'dropout': [0.3, 0.5],
    'batch_size': [32, 64],
    'epochs': [3],
}

TEXTCNN_GRID_STAGE2 = {
    'learning_rate': [0.0005, 0.001, 0.002],
    'embedding_dim': [300],
    'num_filters': [100, 128],
    'dropout': [0.3, 0.4, 0.5],
    'batch_size': [32, 64],
    'epochs': [10],
}

# BiLSTM and LSTM+Attention grids (300d embeddings)
BILSTM_GRID_STAGE1 = {
    'learning_rate': [0.001, 0.01],
    'embedding_dim': [300],
    'hidden_dim': [128],
    'dropout': [0.3, 0.5],
    'batch_size': [32, 64],
    'epochs': [3],
}

BILSTM_GRID_STAGE2 = {
    'learning_rate': [0.0005, 0.001, 0.002],
    'embedding_dim': [300],
    'hidden_dim': [128, 256],
    'dropout': [0.3, 0.4, 0.5],
    'batch_size': [32, 64],
    'epochs': [10],
}

# DistilBERT grids (lower lr, lower dropout, smaller batch)
DISTILBERT_GRID_STAGE1 = {
    'learning_rate': [2e-5, 5e-5],
    'dropout': [0.1],
    'batch_size': [16],
    'epochs': [3],
}

DISTILBERT_GRID_STAGE2 = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'dropout': [0.1, 0.2],
    'batch_size': [16],
    'epochs': [5],
}


def memory_cleanup():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_fashion_mnist_data(tiny_size=500, small_size=5000, full_size=None):
    """
    Load Fashion MNIST data with different subset sizes

    Args:
        tiny_size: Size for stage 1 (default: 500)
        small_size: Size for stage 2 (default: 5000)
        full_size: Size for full training (default: None = all data)

    Returns:
        Dictionary with 'tiny', 'small', 'full' keys containing (X_train, y_train, X_test, y_test)
    """
    logger.info("Loading Fashion MNIST dataset...")
    loader = FashionMNISTLoader(flatten=False, normalize=True)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    # Reshape for CNN: (N, 784) -> (N, 1, 28, 28)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_val = X_val.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    # Combine train + val for CV
    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])

    logger.info(f"Full dataset: {X_full.shape}, Test: {X_test.shape}")

    data = {}

    # Tiny subset
    if tiny_size:
        indices = np.random.choice(len(X_full), min(tiny_size, len(X_full)), replace=False)
        data['tiny'] = (X_full[indices], y_full[indices], X_test[:tiny_size//5], y_test[:tiny_size//5])
        logger.info(f"Tiny subset: Train={len(indices)}, Test={tiny_size//5}")

    # Small subset
    if small_size:
        indices = np.random.choice(len(X_full), min(small_size, len(X_full)), replace=False)
        data['small'] = (X_full[indices], y_full[indices], X_test[:small_size//5], y_test[:small_size//5])
        logger.info(f"Small subset: Train={len(indices)}, Test={small_size//5}")

    # Full dataset
    if full_size is None or full_size >= len(X_full):
        data['full'] = (X_full, y_full, X_test, y_test)
        logger.info(f"Full dataset: Train={len(X_full)}, Test={len(X_test)}")

    return data


def prepare_agnews_data(tiny_size=1000, small_size=10000, full_size=None, max_vocab_size=10000, max_seq_length=200):
    """
    Load AG News data with different subset sizes (tokenized for LSTM)

    Args:
        tiny_size: Size for stage 1 (default: 1000)
        small_size: Size for stage 2 (default: 10000)
        full_size: Size for full training (default: None = all data)
        max_vocab_size: Max vocabulary size (default: 10000)
        max_seq_length: Max sequence length (default: 200)

    Returns:
        Dictionary with 'tiny', 'small', 'full' keys and vocab_size
    """
    logger.info(f"Loading AG News dataset (max_vocab_size={max_vocab_size}, max_seq_length={max_seq_length})...")
    loader = AGNewsTokenizedLoader(max_vocab_size=max_vocab_size, max_seq_length=max_seq_length)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    # Combine train + val
    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])

    logger.info(f"Full dataset: {X_full.shape}, Test: {X_test.shape}")

    data = {}
    data['vocab_size'] = loader.vocab_size

    # Tiny subset
    if tiny_size:
        indices = np.random.choice(len(X_full), min(tiny_size, len(X_full)), replace=False)
        data['tiny'] = (X_full[indices], y_full[indices], X_test[:tiny_size//5], y_test[:tiny_size//5])
        logger.info(f"Tiny subset: Train={len(indices)}, Test={tiny_size//5}")

    # Small subset
    if small_size:
        indices = np.random.choice(len(X_full), min(small_size, len(X_full)), replace=False)
        data['small'] = (X_full[indices], y_full[indices], X_test[:small_size//5], y_test[:small_size//5])
        logger.info(f"Small subset: Train={len(indices)}, Test={small_size//5}")

    # Full dataset
    if full_size is None or full_size >= len(X_full):
        data['full'] = (X_full, y_full, X_test, y_test)
        logger.info(f"Full dataset: Train={len(X_full)}, Test={len(X_test)}")

    return data


def create_model(dataset, model_type, **kwargs):
    """
    Create a PyTorch model

    Args:
        dataset: 'fashion_mnist' or 'ag_news'
        model_type: 'cnn', 'lenet', 'resnet', 'mobilenet', or 'lstm'
        **kwargs: Model-specific parameters

    Returns:
        PyTorchClassifier instance
    """
    if dataset == 'fashion_mnist':
        dropout = kwargs.get('dropout', 0.5)
        learning_rate = kwargs.get('learning_rate', 0.001)
        batch_size = kwargs.get('batch_size', 64)
        epochs = kwargs.get('epochs', 10)
        log_to_mlflow = kwargs.get('log_to_mlflow', False)

        # Select model architecture
        if model_type == 'cnn':
            model = FashionMNISTCNN(
                conv_channels=kwargs.get('conv_channels', (32, 64)),
                kernel_size=kwargs.get('kernel_size', 3),
                use_batchnorm=kwargs.get('use_batchnorm', False),
                fc_units=kwargs.get('fc_units', 128),
                dropout=dropout,
                padding=kwargs.get('padding', 'same'),
            )
        elif model_type == 'lenet':
            model = LeNet5Modified(dropout=dropout)
        elif model_type == 'resnet':
            model = ResNet18Adapted(dropout=dropout)
        elif model_type == 'mobilenet':
            # MobileNet uses lower dropout by default
            dropout = kwargs.get('dropout', 0.2)
            model = MobileNetV2Small(dropout=dropout)
        else:
            raise ValueError(f"Unsupported model type for Fashion MNIST: {model_type}")

        classifier = PyTorchClassifier(
            model=model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            verbose=False,
            log_to_mlflow=log_to_mlflow
        )
        return classifier

    elif dataset == 'ag_news':
        vocab_size = kwargs.get('vocab_size', 10001)
        learning_rate = kwargs.get('learning_rate', 0.001)
        batch_size = kwargs.get('batch_size', 64)
        epochs = kwargs.get('epochs', 10)
        dropout = kwargs.get('dropout', 0.5)
        log_to_mlflow = kwargs.get('log_to_mlflow', False)

        # Select model architecture
        if model_type == 'lstm':
            # Original basic LSTM
            embedding_dim = kwargs.get('embedding_dim', 100)
            hidden_dim = kwargs.get('hidden_dim', 128)
            bidirectional = kwargs.get('bidirectional', False)

            model = AGNewsLSTM(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                bidirectional=bidirectional
            )

        elif model_type == 'textcnn':
            # TextCNN (Kim 2014)
            embedding_dim = kwargs.get('embedding_dim', 300)
            num_filters = kwargs.get('num_filters', 100)

            model = TextCNN(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_filters=num_filters,
                dropout=dropout
            )

        elif model_type == 'bilstm':
            # Simple BiLSTM
            embedding_dim = kwargs.get('embedding_dim', 300)
            hidden_dim = kwargs.get('hidden_dim', 128)

            model = BiLSTMSimple(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )

        elif model_type == 'lstm_attn':
            # LSTM + Attention
            embedding_dim = kwargs.get('embedding_dim', 300)
            hidden_dim = kwargs.get('hidden_dim', 128)

            model = LSTMAttention(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )

        elif model_type == 'distilbert':
            # DistilBERT (requires transformers)
            # Note: DistilBERT has its own tokenizer, not using vocab_size
            dropout = kwargs.get('dropout', 0.1)  # Lower dropout for BERT

            model = DistilBERTClassifier(dropout=dropout)

        else:
            raise ValueError(f"Unsupported model type for AG News: {model_type}")

        classifier = PyTorchClassifier(
            model=model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            verbose=False,
            use_long_tensor=True,  # Text models need token indices as LongTensor
            log_to_mlflow=log_to_mlflow
        )
        return classifier

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def cross_validate(X, y, dataset, model_type, cv_folds=3, **params):
    """
    Perform cross-validation

    Args:
        X: Training data
        y: Training labels
        dataset: Dataset name
        model_type: Model type
        cv_folds: Number of CV folds (default: 3)
        **params: Model hyperparameters

    Returns:
        Mean CV accuracy, std CV accuracy
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Create and train model
        classifier = create_model(dataset, model_type, **params)
        classifier.fit(X_train_fold, y_train_fold)

        # Evaluate
        y_pred = classifier.predict(X_val_fold)
        accuracy = (y_pred == y_val_fold).mean()
        scores.append(accuracy)

        # Clean up
        del classifier
        memory_cleanup()

    return np.mean(scores), np.std(scores)


def grid_search(X, y, dataset, model_type, param_grid, cv_folds=3, experiment_name=None):
    """
    Perform grid search with cross-validation

    Args:
        X: Training data
        y: Training labels
        dataset: Dataset name
        model_type: Model type
        param_grid: Dictionary of hyperparameter lists
        cv_folds: Number of CV folds (default: 3)
        experiment_name: MLflow experiment name

    Returns:
        best_params, best_score, results
    """
    # Set up MLflow
    if experiment_name:
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        mlflow.set_experiment(experiment_name)

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    logger.info(f"Grid search: {len(param_combinations)} combinations, {cv_folds}-fold CV")

    results = []
    best_score = 0
    best_params = None

    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))

        # Extract vocab_size if AG News (it might be in params as a list)
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

        # Track best
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            logger.info(f"  *** New best score: {best_score:.4f}")

        # Clean up
        memory_cleanup()

    logger.info(f"\n{'='*80}")
    logger.info(f"Grid Search Complete!")
    logger.info(f"Best CV Accuracy: {best_score:.4f}")
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"{'='*80}")

    return best_params, best_score, results


def train_final_model(X_train, y_train, X_test, y_test, dataset, model_type, params, experiment_name=None):
    """
    Train final model on full training set and evaluate on test set

    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        dataset: Dataset name
        model_type: Model type
        params: Hyperparameters
        experiment_name: MLflow experiment name

    Returns:
        test_accuracy, model
    """
    logger.info(f"\nTraining final model with best parameters...")
    logger.info(f"Parameters: {params}")

    # Set up MLflow
    if experiment_name:
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        mlflow.set_experiment(experiment_name)

    # Create model with MLflow logging enabled (fit logs per-epoch metrics)
    params_with_mlflow = params.copy()
    params_with_mlflow['log_to_mlflow'] = True
    classifier = create_model(dataset, model_type, **params_with_mlflow)

    active_run = mlflow.active_run()
    use_active = False
    if experiment_name:
        run_ctx = mlflow.start_run(run_name=f"{model_type}_final")
    elif active_run:
        run_ctx = None
        use_active = True
    else:
        run_ctx = None

    if run_ctx:
        with run_ctx:
            start_time = time.time()
            classifier.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_pred = classifier.predict(X_test)
            test_accuracy = (y_pred == y_test).mean()

            mlflow.log_params(params)
            mlflow.log_metric('test_accuracy', test_accuracy)
            mlflow.log_metric('train_time', train_time)

            # Save model
            try:
                mlflow.pytorch.log_model(classifier.model, "model")
            except Exception as e:
                logger.warning(f"Could not log PyTorch model: {e}")
    else:
        start_time = time.time()
        classifier.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = classifier.predict(X_test)
        test_accuracy = (y_pred == y_test).mean()

        if use_active:
            mlflow.log_params(params)
            mlflow.log_metric('test_accuracy', test_accuracy)
            mlflow.log_metric('train_time', train_time)
            try:
                mlflow.pytorch.log_model(classifier.model, "model")
            except Exception as e:
                logger.warning(f"Could not log PyTorch model: {e}")

    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Training time: {train_time:.1f}s")

    return test_accuracy, classifier


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train deep learning models with memory-efficient hyperparameter tuning"
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
        required=True,
        choices=['cnn', 'lenet', 'resnet', 'mobilenet', 'lstm', 'textcnn', 'bilstm', 'lstm_attn', 'distilbert'],
        help='Model type - Fashion MNIST: cnn, lenet, resnet, mobilenet | AG News: lstm, textcnn, bilstm, lstm_attn, distilbert'
    )

    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2],
        help='Stage: 1=Tiny grid search, 2=Small refined search'
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Run both stages + final training'
    )

    parser.add_argument(
        '--cv-folds',
        type=int,
        default=3,
        help='Number of CV folds (default: 3)'
    )

    args = parser.parse_args()

    # Validation
    if not args.stage and not args.full:
        parser.error("Must specify --stage or --full")

    if args.dataset == 'fashion_mnist' and args.model not in ['cnn', 'lenet', 'resnet', 'mobilenet']:
        parser.error("Fashion MNIST requires cnn, lenet, resnet, or mobilenet model")

    if args.dataset == 'ag_news' and args.model not in ['lstm', 'textcnn', 'bilstm', 'lstm_attn', 'distilbert']:
        parser.error("AG News requires lstm, textcnn, bilstm, lstm_attn, or distilbert model")

    return args


def main():
    """Main training pipeline"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info(f"DEEP LEARNING PIPELINE: {args.dataset.upper()} - {args.model.upper()}")
    logger.info("=" * 80)

    # Load data and select hyperparameter grids
    if args.dataset == 'fashion_mnist':
        data = prepare_fashion_mnist_data()

        # Select grid based on model type
        if args.model == 'resnet':
            stage1_grid = RESNET_GRID_STAGE1
            stage2_grid = RESNET_GRID_STAGE2
        elif args.model == 'mobilenet':
            stage1_grid = MOBILENET_GRID_STAGE1
            stage2_grid = MOBILENET_GRID_STAGE2
        else:  # cnn, lenet
            stage1_grid = FASHION_MNIST_GRID_STAGE1
            stage2_grid = FASHION_MNIST_GRID_STAGE2
    else:  # ag_news
        data = prepare_agnews_data()

        # Select grid based on model type
        if args.model == 'textcnn':
            stage1_grid = TEXTCNN_GRID_STAGE1.copy()
            stage2_grid = TEXTCNN_GRID_STAGE2.copy()
        elif args.model in ['bilstm', 'lstm_attn']:
            stage1_grid = BILSTM_GRID_STAGE1.copy()
            stage2_grid = BILSTM_GRID_STAGE2.copy()
        elif args.model == 'distilbert':
            stage1_grid = DISTILBERT_GRID_STAGE1.copy()
            stage2_grid = DISTILBERT_GRID_STAGE2.copy()
        else:  # lstm
            stage1_grid = AG_NEWS_GRID_STAGE1.copy()
            stage2_grid = AG_NEWS_GRID_STAGE2.copy()

        # Add vocab_size to grids (not needed for DistilBERT)
        if args.model != 'distilbert':
            stage1_grid['vocab_size'] = [data['vocab_size']]
            stage2_grid['vocab_size'] = [data['vocab_size']]

    # Determine stages
    stages = [1, 2] if args.full else [args.stage]

    best_params_stage1 = None
    best_params_stage2 = None

    # STAGE 1: Tiny grid search
    if 1 in stages:
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: TINY GRID SEARCH (Find Important Hyperparameters)")
        logger.info("=" * 80)

        X_train, y_train, X_test, y_test = data['tiny']
        experiment_name = f"{args.dataset}_{args.model}_stage1_tiny"

        best_params_stage1, best_score, _ = grid_search(
            X_train, y_train,
            args.dataset, args.model,
            stage1_grid,
            cv_folds=min(3, args.cv_folds),  # Use fewer folds for tiny data
            experiment_name=experiment_name
        )

        logger.info(f"\n✓ Stage 1 complete - Best CV: {best_score:.4f}")

    # STAGE 2: Small refined search
    if 2 in stages:
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: SMALL REFINED SEARCH (Fine-tune Hyperparameters)")
        logger.info("=" * 80)

        X_train, y_train, X_test, y_test = data['small']
        experiment_name = f"{args.dataset}_{args.model}_stage2_small"

        # Use refined grid with 2-fold CV for speed (was 3-fold)
        # Stage 2 focuses on refinement, so 2-fold CV is sufficient
        best_params_stage2, best_score, _ = grid_search(
            X_train, y_train,
            args.dataset, args.model,
            stage2_grid,
            cv_folds=min(2, args.cv_folds),  # Use 2-fold CV for faster Stage 2
            experiment_name=experiment_name
        )

        logger.info(f"\n✓ Stage 2 complete - Best CV: {best_score:.4f}")

    # FULL TRAINING (only if --full flag)
    if args.full:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL TRAINING: Full Dataset with Best Hyperparameters")
        logger.info("=" * 80)

        X_train, y_train, X_test, y_test = data['full']
        experiment_name = f"{args.dataset}_{args.model}_final"

        # Use best params from stage 2 (or stage 1 if stage 2 not run)
        best_params = best_params_stage2 or best_params_stage1

        if best_params:
            # Increase epochs for final training
            final_params = best_params.copy()
            final_params['epochs'] = 20  # More epochs for final model

            test_acc, model = train_final_model(
                X_train, y_train, X_test, y_test,
                args.dataset, args.model,
                final_params,
                experiment_name=experiment_name
            )

            logger.info(f"\n✓ Final training complete - Test Accuracy: {test_acc:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nTo view results in MLflow UI:")
    logger.info("  mlflow ui")
    logger.info("  Open: http://localhost:5000")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nError: {e}", exc_info=True)
        sys.exit(1)
