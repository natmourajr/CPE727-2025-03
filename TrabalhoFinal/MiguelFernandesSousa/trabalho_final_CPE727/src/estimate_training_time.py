"""
Training Time Estimation Script

This script estimates how long each model will take to train by running
a tiny subset (100 samples, 1 epoch) and extrapolating the time.

Usage:
    uv run python src/estimate_training_time.py
    uv run python src/estimate_training_time.py --dataset fashion_mnist
    uv run python src/estimate_training_time.py --dataset ag_news
"""
import argparse
import time
import sys
from pathlib import Path

import numpy as np
import torch

from src.data_loader import FashionMNISTLoader
from src.data_loader_agnews_tokenized import AGNewsTokenizedLoader
from src.models_deep import (
    FashionMNISTCNN, LeNet5Modified, ResNet18Adapted, MobileNetV2Small,
    AGNewsLSTM, TextCNN, BiLSTMSimple, LSTMAttention, DistilBERTClassifier,
    PyTorchClassifier, TRANSFORMERS_AVAILABLE
)
from src.utils import setup_logger

logger = setup_logger(__name__)

# Dataset sizes for extrapolation
FASHION_MNIST_SIZES = {
    'tiny': 500,
    'small': 5000,
    'full': 60000,
}

AG_NEWS_SIZES = {
    'tiny': 1000,
    'small': 10000,
    'full': 120000,
}

# Epochs per stage
EPOCHS = {
    'stage1': 3,
    'stage2': 10,
    'final': 20,
}

# Fashion MNIST models
FASHION_MNIST_MODELS = {
    'cnn': (FashionMNISTCNN, {'dropout': 0.5}),
    'lenet': (LeNet5Modified, {'dropout': 0.5}),
    'resnet': (ResNet18Adapted, {'dropout': 0.5}),
    'mobilenet': (MobileNetV2Small, {'dropout': 0.2}),
}

# AG News models
AG_NEWS_MODELS = {
    'lstm': (AGNewsLSTM, {'vocab_size': 10001, 'embedding_dim': 100, 'hidden_dim': 128, 'dropout': 0.5, 'bidirectional': False}),
    'textcnn': (TextCNN, {'vocab_size': 10001, 'embedding_dim': 300, 'num_filters': 100, 'dropout': 0.5}),
    'bilstm': (BiLSTMSimple, {'vocab_size': 10001, 'embedding_dim': 300, 'hidden_dim': 128, 'dropout': 0.5}),
    'lstm_attn': (LSTMAttention, {'vocab_size': 10001, 'embedding_dim': 300, 'hidden_dim': 128, 'dropout': 0.5}),
}

if TRANSFORMERS_AVAILABLE:
    AG_NEWS_MODELS['distilbert'] = (DistilBERTClassifier, {'dropout': 0.1})


def load_tiny_data_fashion_mnist(n_samples=100):
    """Load tiny subset of Fashion MNIST for estimation"""
    logger.info(f"Loading {n_samples} Fashion MNIST samples...")
    loader = FashionMNISTLoader(flatten=False, normalize=True)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    # Reshape for CNN
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    # Take tiny subset
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_tiny = X_train[indices]
    y_tiny = y_train[indices]

    # Small test set
    test_indices = np.random.choice(len(X_test), 100, replace=False)
    X_test_tiny = X_test[test_indices]
    y_test_tiny = y_test[test_indices]

    return X_tiny, y_tiny, X_test_tiny, y_test_tiny


def load_tiny_data_agnews(n_samples=100):
    """Load tiny subset of AG News for estimation"""
    logger.info(f"Loading {n_samples} AG News samples...")
    loader = AGNewsTokenizedLoader(max_vocab_size=10000, max_seq_length=200)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    vocab_size = loader.vocab_size

    # Take tiny subset
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_tiny = X_train[indices]
    y_tiny = y_train[indices]

    # Small test set
    test_indices = np.random.choice(len(X_test), 100, replace=False)
    X_test_tiny = X_test[test_indices]
    y_test_tiny = y_test[test_indices]

    return X_tiny, y_tiny, X_test_tiny, y_test_tiny, vocab_size


def estimate_model_time(model_class, model_params, X_train, y_train, X_test, y_test,
                       batch_size=32, use_long_tensor=False):
    """
    Estimate training time for a model by running 1 epoch on tiny dataset

    Returns:
        time_per_epoch (seconds)
    """
    # Create model
    model = model_class(**model_params)

    # Create classifier
    classifier = PyTorchClassifier(
        model=model,
        learning_rate=0.001,
        batch_size=batch_size,
        epochs=1,  # Just 1 epoch for estimation
        verbose=False,
        use_long_tensor=use_long_tensor,
        log_to_mlflow=False
    )

    # Time the training
    start_time = time.time()
    classifier.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    # Quick test to ensure it works
    try:
        y_pred = classifier.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        logger.debug(f"  Quick test accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.warning(f"  Test failed: {e}")

    # Clean up
    del classifier
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return elapsed_time


def extrapolate_times(time_per_epoch_tiny, n_tiny, sizes, epochs):
    """
    Extrapolate training times based on tiny subset

    Args:
        time_per_epoch_tiny: Time for 1 epoch on n_tiny samples
        n_tiny: Number of samples in tiny subset
        sizes: Dict of dataset sizes {'tiny': X, 'small': Y, 'full': Z}
        epochs: Dict of epochs per stage

    Returns:
        Dict with estimated times
    """
    estimates = {}

    for stage, n_samples in sizes.items():
        # Linear scaling (conservative estimate)
        scale_factor = n_samples / n_tiny

        # Stage 1
        stage1_time = time_per_epoch_tiny * scale_factor * epochs['stage1']
        estimates[f'{stage}_stage1'] = stage1_time

        # Stage 2
        stage2_time = time_per_epoch_tiny * scale_factor * epochs['stage2']
        estimates[f'{stage}_stage2'] = stage2_time

        # Final
        final_time = time_per_epoch_tiny * scale_factor * epochs['final']
        estimates[f'{stage}_final'] = final_time

        # Full pipeline
        estimates[f'{stage}_full_pipeline'] = stage1_time + stage2_time + final_time

    return estimates


def format_time(seconds):
    """Format seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_fashion_mnist():
    """Estimate times for all Fashion MNIST models"""
    logger.info("=" * 80)
    logger.info("FASHION MNIST - TRAINING TIME ESTIMATION")
    logger.info("=" * 80)

    # Load tiny data
    X_tiny, y_tiny, X_test_tiny, y_test_tiny = load_tiny_data_fashion_mnist(n_samples=100)
    n_tiny = len(X_tiny)

    results = {}

    for model_name, (model_class, model_params) in FASHION_MNIST_MODELS.items():
        logger.info(f"\nEstimating: {model_name.upper()}")
        logger.info("-" * 40)

        try:
            # Estimate time per epoch
            time_per_epoch = estimate_model_time(
                model_class, model_params,
                X_tiny, y_tiny, X_test_tiny, y_test_tiny,
                batch_size=32,
                use_long_tensor=False
            )

            logger.info(f"  Time for 1 epoch on {n_tiny} samples: {format_time(time_per_epoch)}")

            # Extrapolate
            estimates = extrapolate_times(
                time_per_epoch, n_tiny,
                FASHION_MNIST_SIZES,
                EPOCHS
            )

            results[model_name] = estimates

            # Print key estimates
            logger.info(f"  Estimated times:")
            logger.info(f"    Stage 1 (tiny, {FASHION_MNIST_SIZES['tiny']} samples, {EPOCHS['stage1']} epochs): {format_time(estimates['tiny_stage1'])}")
            logger.info(f"    Stage 2 (small, {FASHION_MNIST_SIZES['small']} samples, {EPOCHS['stage2']} epochs): {format_time(estimates['small_stage2'])}")
            logger.info(f"    Final (full, {FASHION_MNIST_SIZES['full']} samples, {EPOCHS['final']} epochs): {format_time(estimates['full_final'])}")
            logger.info(f"    FULL PIPELINE: {format_time(estimates['tiny_full_pipeline'])} (stage1) + {format_time(estimates['small_full_pipeline'])} (stage2) + {format_time(estimates['full_final'])} (final) = {format_time(estimates['tiny_stage1'] + estimates['small_stage2'] + estimates['full_final'])}")

        except Exception as e:
            logger.error(f"  Failed to estimate {model_name}: {e}")
            results[model_name] = None

    return results


def estimate_agnews():
    """Estimate times for all AG News models"""
    logger.info("=" * 80)
    logger.info("AG NEWS - TRAINING TIME ESTIMATION")
    logger.info("=" * 80)

    # Load tiny data
    X_tiny, y_tiny, X_test_tiny, y_test_tiny, vocab_size = load_tiny_data_agnews(n_samples=100)
    n_tiny = len(X_tiny)

    results = {}

    for model_name, (model_class, model_params) in AG_NEWS_MODELS.items():
        logger.info(f"\nEstimating: {model_name.upper()}")
        logger.info("-" * 40)

        # Update vocab_size
        params = model_params.copy()
        if 'vocab_size' in params:
            params['vocab_size'] = vocab_size

        # Adjust batch size for DistilBERT
        batch_size = 16 if model_name == 'distilbert' else 32

        try:
            # Estimate time per epoch
            time_per_epoch = estimate_model_time(
                model_class, params,
                X_tiny, y_tiny, X_test_tiny, y_test_tiny,
                batch_size=batch_size,
                use_long_tensor=True
            )

            logger.info(f"  Time for 1 epoch on {n_tiny} samples: {format_time(time_per_epoch)}")

            # Extrapolate
            estimates = extrapolate_times(
                time_per_epoch, n_tiny,
                AG_NEWS_SIZES,
                EPOCHS
            )

            results[model_name] = estimates

            # Print key estimates
            logger.info(f"  Estimated times:")
            logger.info(f"    Stage 1 (tiny, {AG_NEWS_SIZES['tiny']} samples, {EPOCHS['stage1']} epochs): {format_time(estimates['tiny_stage1'])}")
            logger.info(f"    Stage 2 (small, {AG_NEWS_SIZES['small']} samples, {EPOCHS['stage2']} epochs): {format_time(estimates['small_stage2'])}")
            logger.info(f"    Final (full, {AG_NEWS_SIZES['full']} samples, {EPOCHS['final']} epochs): {format_time(estimates['full_final'])}")
            logger.info(f"    FULL PIPELINE: {format_time(estimates['tiny_full_pipeline'])} (stage1) + {format_time(estimates['small_full_pipeline'])} (stage2) + {format_time(estimates['full_final'])} (final) = {format_time(estimates['tiny_stage1'] + estimates['small_stage2'] + estimates['full_final'])}")

        except Exception as e:
            logger.error(f"  Failed to estimate {model_name}: {e}")
            results[model_name] = None

    return results


def print_summary_table(fashion_mnist_results, agnews_results):
    """Print summary table comparing all models"""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - ESTIMATED FULL PIPELINE TIMES (Stage 1 + Stage 2 + Final)")
    logger.info("=" * 80)

    # Fashion MNIST
    logger.info("\nFashion MNIST:")
    logger.info("-" * 60)
    logger.info(f"{'Model':<15} {'Stage 1':<12} {'Stage 2':<12} {'Final':<12} {'Total':<12}")
    logger.info("-" * 60)

    for model_name, estimates in fashion_mnist_results.items():
        if estimates:
            stage1 = estimates['tiny_stage1']
            stage2 = estimates['small_stage2']
            final = estimates['full_final']
            total = stage1 + stage2 + final

            logger.info(f"{model_name:<15} {format_time(stage1):<12} {format_time(stage2):<12} {format_time(final):<12} {format_time(total):<12}")

    # AG News
    logger.info("\nAG News:")
    logger.info("-" * 60)
    logger.info(f"{'Model':<15} {'Stage 1':<12} {'Stage 2':<12} {'Final':<12} {'Total':<12}")
    logger.info("-" * 60)

    for model_name, estimates in agnews_results.items():
        if estimates:
            stage1 = estimates['tiny_stage1']
            stage2 = estimates['small_stage2']
            final = estimates['full_final']
            total = stage1 + stage2 + final

            logger.info(f"{model_name:<15} {format_time(stage1):<12} {format_time(stage2):<12} {format_time(final):<12} {format_time(total):<12}")


def main():
    """Main estimation function"""
    parser = argparse.ArgumentParser(
        description="Estimate training times for deep learning models"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['fashion_mnist', 'ag_news', 'both'],
        default='both',
        help='Which dataset to estimate (default: both)'
    )
    args = parser.parse_args()

    # Check device
    if torch.backends.mps.is_available():
        device = "MPS (Metal Performance Shaders)"
    elif torch.cuda.is_available():
        device = "CUDA"
    else:
        device = "CPU"

    logger.info(f"\nDevice: {device}")
    logger.info("Note: Estimates are conservative linear extrapolations")
    logger.info("Actual times may vary based on hardware and system load\n")

    fashion_mnist_results = {}
    agnews_results = {}

    if args.dataset in ['fashion_mnist', 'both']:
        fashion_mnist_results = estimate_fashion_mnist()

    if args.dataset in ['ag_news', 'both']:
        agnews_results = estimate_agnews()

    if args.dataset == 'both':
        print_summary_table(fashion_mnist_results, agnews_results)

    logger.info("\n" + "=" * 80)
    logger.info("ESTIMATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nUse these estimates to plan your training schedule!")
    logger.info("\nTo train a model:")
    logger.info("  uv run python src/train_deep.py --dataset <dataset> --model <model> --full")


if __name__ == "__main__":
    main()
