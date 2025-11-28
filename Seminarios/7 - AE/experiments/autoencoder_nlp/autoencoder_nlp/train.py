#!/usr/bin/env python3
"""
Training script for the Autoencoder model on RCV1 dataset.
"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime
import platform
import yaml
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from .model import Autoencoder
from .rcv1_dataset import RCV1Dataset, collate_fn


class HDF5MetricsLogger(Callback):
    """
    Custom Lightning callback to log all training metrics to an HDF5 file.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the HDF5 metrics logger.

        Args:
            output_dir: Directory where the metrics HDF5 file will be saved
        """
        super().__init__()
        self.output_dir = output_dir
        self.metrics_file = output_dir / "training_metrics.h5"
        self.metrics = {
            "epoch": [],
            "step": [],
            "train_loss": [],
            "train_recon_loss": [],
            "train_l1_loss": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_l1_loss": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""
        # Get logged metrics
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Store epoch-level metrics
        self.metrics["epoch"].append(epoch)
        self.metrics["step"].append(trainer.global_step)

        # Store training metrics (use last logged values)
        for key in ["train_loss", "train_recon_loss", "train_l1_loss"]:
            if key in metrics:
                self.metrics[key].append(metrics[key].item())
            else:
                self.metrics[key].append(np.nan)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        metrics = trainer.callback_metrics

        # Store validation metrics
        for key in ["val_loss", "val_recon_loss", "val_l1_loss"]:
            if key in metrics:
                self.metrics[key].append(metrics[key].item())
            else:
                self.metrics[key].append(np.nan)

    def on_train_end(self, trainer, pl_module):
        """Called when training ends. Save all metrics to HDF5."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.metrics_file, "w") as f:
            for key, values in self.metrics.items():
                if values:  # Only save non-empty lists
                    f.create_dataset(key, data=np.array(values))

        print(f"Training metrics saved to {self.metrics_file}")


def save_config(
    config_path: Path,
    dataset_path: Path,
    output_dir: Path,
    dimensions: List[int],
    mlflow_tracking_uri: str,
    activation: str,
    latent_activation: str,
    l1_alpha: Optional[float],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    val_split: float,
    num_workers: int,
    seed: int,
    precision: str,
    num_features: int,
    train_size: int,
    val_size: int,
) -> None:
    """
    Save training configuration to YAML file.

    Args:
        config_path: Path where the config YAML will be saved
        (other args): Training configuration parameters
    """
    config = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "num_features": num_features,
            "total_samples": train_size + val_size,
            "train_samples": train_size,
            "val_samples": val_size,
            "val_split": val_split,
        },
        "paths": {
            "output_dir": str(output_dir),
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "metrics_file": str(output_dir / "training_metrics.h5"),
        },
        "model": {
            "dimensions": dimensions,
            "activation": activation,
            "latent_activation": latent_activation,
            "l1_alpha": l1_alpha,
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        "training": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_workers": num_workers,
            "seed": seed,
            "precision": precision,
        },
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "lightning_version": L.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to {config_path}")


def train(
    dataset_path: Path,
    output_dir: Path,
    dimensions: List[int],
    mlflow_tracking_uri: Optional[str] = None,
    activation: str = "relu",
    latent_activation: str = "linear",
    l1_alpha: Optional[float] = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    num_epochs: int = 100,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    precision: str = "32-true",
) -> None:
    """
    Train the Autoencoder model on RCV1 dataset.

    Args:
        dataset_path: Path to the RCV1 HDF5 dataset file
        output_dir: Directory where outputs (checkpoints, metrics) will be saved
        dimensions: List of layer dimensions for the autoencoder
        mlflow_tracking_uri: MLflow tracking URI or directory (default: ./mlruns)
        activation: Activation function for intermediate layers
        latent_activation: Activation function for latent space
        l1_alpha: L1 regularization coefficient for latent activations
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: Weight decay for AdamW optimizer
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        num_workers: Number of DataLoader workers (use 0 for HDF5)
        seed: Random seed for reproducibility
        precision: Training precision. Options: "32-true" (float32), "bf16-mixed" (bfloat16 mixed),
                  "16-mixed" (float16 mixed). For RTX 4070, use "bf16-mixed" for best performance.
    """
    # Set random seed for reproducibility
    L.seed_everything(seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking URI default
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = "./mlruns"

    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"MLflow tracking: {mlflow_tracking_uri}")
    print(f"Model dimensions: {dimensions}")
    print(f"Activation: {activation}")
    print(f"Latent activation: {latent_activation}")
    print(f"L1 alpha: {l1_alpha}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Validation split: {val_split}")
    print(f"Precision: {precision}")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = RCV1Dataset(dataset_path)
    print(f"Total samples: {len(full_dataset)}")
    print(f"Number of features: {full_dataset.num_features}")

    # Verify dimensions match
    if dimensions[0] != full_dataset.num_features:
        raise ValueError(
            f"First dimension ({dimensions[0]}) must match dataset features ({full_dataset.num_features})"
        )

    # Train-validation split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Save configuration to YAML
    print("\nSaving configuration...")
    config_path = output_dir / "config.yaml"
    save_config(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        dimensions=dimensions,
        mlflow_tracking_uri=mlflow_tracking_uri,
        activation=activation,
        latent_activation=latent_activation,
        l1_alpha=l1_alpha,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_epochs=num_epochs,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
        precision=precision,
        num_features=full_dataset.num_features,
        train_size=train_size,
        val_size=val_size,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=False,
    )

    # Initialize model
    print("\nInitializing model...")
    model = Autoencoder(
        dimensions=dimensions,
        activation=activation,
        latent_activation=latent_activation,
        l1_alpha=l1_alpha,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Set up MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name="autoencoder_nlp",
        save_dir=mlflow_tracking_uri,
        log_model=True,
    )

    # Set up HDF5 metrics logger
    hdf5_logger = HDF5MetricsLogger(output_dir)

    # Set up ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="autoencoder-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,  # Save top 3 best models
        save_last=True,  # Also save the last checkpoint
        verbose=True,
    )

    # Create trainer
    print("\nSetting up trainer...")
    trainer = L.Trainer(
        max_epochs=num_epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, hdf5_logger],
        accelerator="auto",  # Automatically use GPU if available
        devices="auto",
        precision=precision,  # Training precision (bf16-mixed recommended for RTX 4070)
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train the model
    print("\nStarting training...")
    print("=" * 80)
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"MLflow logs saved to: {mlflow_tracking_uri}")
    print(f"Metrics saved to: {output_dir / 'training_metrics.h5'}")
    print(f"Configuration saved to: {output_dir / 'config.yaml'}")
    print("=" * 80)

    # Close dataset
    full_dataset.close()
