import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict
import lightning as L
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import yaml
import os

from .models import RNNSiamese, CNNSiamese

# Disable tokenizers parallelism to avoid warnings with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_fn(batch):
    """Collate function to pad sequences in a batch from HuggingFace dataset."""
    sentence1_ids = [torch.tensor(item['sentence1_ids'], dtype=torch.long) for item in batch]
    sentence2_ids = [torch.tensor(item['sentence2_ids'], dtype=torch.long) for item in batch]
    scores = [item['score'] for item in batch]

    sentence1_padded = torch.nn.utils.rnn.pad_sequence(
        sentence1_ids, batch_first=True, padding_value=0
    )
    sentence2_padded = torch.nn.utils.rnn.pad_sequence(
        sentence2_ids, batch_first=True, padding_value=0
    )
    scores_tensor = torch.tensor(scores, dtype=torch.float)

    return sentence1_padded, sentence2_padded, scores_tensor


class MetricsLogger(L.Callback):
    """Callback to log metrics to HDF5 file."""

    def __init__(self, output_path: Path):
        super().__init__()
        self.output_path = output_path
        self.metrics = {
            'train_loss': [],
            'train_mae': [],
            'train_acc': [],
            'train_f1': [],
            'train_pearson': [],
            'val_loss': [],
            'val_mae': [],
            'val_acc': [],
            'val_f1': [],
            'val_pearson': [],
            'epoch': [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        self.metrics['epoch'].append(epoch)

        for key in self.metrics.keys():
            if key == 'epoch':
                continue
            metric_value = trainer.callback_metrics.get(key, float('nan'))
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            self.metrics[key].append(metric_value)

    def on_fit_end(self, trainer, pl_module):
        output_file = self.output_path / 'training_logs.h5'
        with h5py.File(output_file, 'w') as f:
            for key, values in self.metrics.items():
                f.create_dataset(key, data=np.array(values))
        print(f"Training logs saved to: {output_file}")


def train_rnn(
    train_dataset: DatasetDict,
    val_dataset: DatasetDict,
    n_tokens: int,
    embedding_dim: int = 300,
    n_layers: int = 1,
    n_hidden: int = 128,
    n_fc_hidden: int = 128,
    rnn_type: str = 'lstm',
    dropout: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    bidirectional: bool = True,
    padding_idx: int = 0,
    similarity_threshold: float = 3.0,
    batch_size: int = 32,
    max_epochs: int = 50,
    num_workers: int = 4,
    accelerator: str = 'auto',
    devices: int = 1,
    gradient_clip_val: float = 1.0,
    seed: int = 202512,
    output_dir: str = 'outputs',
    experiment_name: Optional[str] = None,
    checkpoint_monitor: str = 'val_loss',
    checkpoint_mode: str = 'min',
):
    """
    Train a Siamese RNN model for text similarity.

    Parameters
    ----------
    train_dataset : DatasetDict
        HuggingFace dataset for training with columns: sentence1_ids, sentence2_ids, score
    val_dataset : DatasetDict
        HuggingFace dataset for validation with columns: sentence1_ids, sentence2_ids, score
    n_tokens : int
        Vocabulary size
    embedding_dim : int, default=300
        Dimension of word embeddings
    n_layers : int, default=1
        Number of RNN layers
    n_hidden : int, default=128
        Hidden size of RNN
    n_fc_hidden : int, default=128
        Hidden size of fully connected classifier
    rnn_type : str, default='lstm'
        Type of RNN: 'lstm' or 'gru'
    dropout : float, default=0.5
        Dropout rate
    learning_rate : float, default=1e-3
        Learning rate for AdamW optimizer
    weight_decay : float, default=1e-5
        Weight decay for AdamW optimizer
    bidirectional : bool, default=True
        Whether to use bidirectional RNN
    padding_idx : int, default=0
        Padding token index
    similarity_threshold : float, default=3.0
        Threshold for binary classification metrics
    batch_size : int, default=32
        Batch size for training
    max_epochs : int, default=50
        Maximum number of training epochs
    num_workers : int, default=4
        Number of workers for data loading
    accelerator : str, default='auto'
        Accelerator type (auto, gpu, cpu, etc.)
    devices : int, default=1
        Number of devices to use
    gradient_clip_val : float, default=1.0
        Gradient clipping value (important for RNN stability)
    seed : int, default=202512
        Random seed for reproducibility
    output_dir : str, default='outputs'
        Directory to save outputs (checkpoints, logs)
    experiment_name : str or None, default=None
        Name for the experiment (used in logging)
    checkpoint_monitor : str, default='val_loss'
        Metric to monitor for checkpointing
    checkpoint_mode : str, default='min'
        Mode for checkpointing ('min' or 'max')

    Returns
    -------
    RNNSiamese
        Trained model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        'n_tokens': n_tokens,
        'embedding_dim': embedding_dim,
        'n_layers': n_layers,
        'n_hidden': n_hidden,
        'n_fc_hidden': n_fc_hidden,
        'rnn_type': rnn_type,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'bidirectional': bidirectional,
        'padding_idx': padding_idx,
        'similarity_threshold': similarity_threshold,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'num_workers': num_workers,
        'accelerator': accelerator,
        'devices': devices,
        'gradient_clip_val': gradient_clip_val,
        'seed': seed,
        'checkpoint_monitor': checkpoint_monitor,
        'checkpoint_mode': checkpoint_mode,
    }

    hyperparameters_file = output_path / 'hyperparameters.yaml'
    with open(hyperparameters_file, 'w') as f:
        yaml.dump(hyperparameters, f, default_flow_style=False, sort_keys=False)
    print(f"Hyperparameters saved to: {hyperparameters_file}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if accelerator == 'gpu' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if accelerator == 'gpu' else False,
    )

    model = RNNSiamese(
        n_tokens=n_tokens,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_fc_hidden=n_fc_hidden,
        rnn_type=rnn_type,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bidirectional=bidirectional,
        padding_idx=padding_idx,
        similarity_threshold=similarity_threshold,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=output_path / 'checkpoints',
        filename=f'{experiment_name or "rnn"}-{{epoch:02d}}-{{val_loss:.4f}}',
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_top_k=1,
        save_last=True,
    )

    metrics_logger = MetricsLogger(output_path)

    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=output_path / 'logs',
        name=experiment_name or 'rnn',
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, metrics_logger],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=gradient_clip_val,
    )

    trainer.fit(model, train_loader, val_loader)

    return model


def train_cnn(
    train_dataset: DatasetDict,
    val_dataset: DatasetDict,
    n_tokens: int,
    embedding_dim: int = 300,
    kernel_sizes: list = None,
    n_filters: int = 128,
    n_fc_hidden: int = 128,
    dropout: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    padding_idx: int = 0,
    similarity_threshold: float = 3.0,
    pooling_strategy: str = 'max',
    batch_size: int = 32,
    max_epochs: int = 50,
    num_workers: int = 4,
    accelerator: str = 'auto',
    devices: int = 1,
    gradient_clip_val: float = 1.0,
    seed: int = 202512,
    output_dir: str = 'outputs',
    experiment_name: Optional[str] = None,
    checkpoint_monitor: str = 'val_loss',
    checkpoint_mode: str = 'min',
):
    """
    Train a Siamese CNN model for text similarity.

    Parameters
    ----------
    train_dataset : DatasetDict
        HuggingFace dataset for training with columns: sentence1_ids, sentence2_ids, score
    val_dataset : DatasetDict
        HuggingFace dataset for validation with columns: sentence1_ids, sentence2_ids, score
    n_tokens : int
        Vocabulary size
    embedding_dim : int, default=300
        Dimension of word embeddings
    kernel_sizes : list, default=None
        List of kernel sizes for convolutional layers (default: [3, 4, 5])
    n_filters : int, default=128
        Number of filters for each convolutional layer
    n_fc_hidden : int, default=128
        Hidden size of fully connected classifier
    dropout : float, default=0.5
        Dropout rate
    learning_rate : float, default=1e-3
        Learning rate for AdamW optimizer
    weight_decay : float, default=1e-5
        Weight decay for AdamW optimizer
    padding_idx : int, default=0
        Padding token index
    similarity_threshold : float, default=3.0
        Threshold for binary classification metrics
    pooling_strategy : str, default='max'
        Pooling strategy: 'max', 'mean', or 'both'
    batch_size : int, default=32
        Batch size for training
    max_epochs : int, default=50
        Maximum number of training epochs
    num_workers : int, default=4
        Number of workers for data loading
    accelerator : str, default='auto'
        Accelerator type (auto, gpu, cpu, etc.)
    devices : int, default=1
        Number of devices to use
    gradient_clip_val : float, default=1.0
        Gradient clipping value
    seed : int, default=202512
        Random seed for reproducibility
    output_dir : str, default='outputs'
        Directory to save outputs (checkpoints, logs)
    experiment_name : str or None, default=None
        Name for the experiment (used in logging)
    checkpoint_monitor : str, default='val_loss'
        Metric to monitor for checkpointing
    checkpoint_mode : str, default='min'
        Mode for checkpointing ('min' or 'max')

    Returns
    -------
    CNNSiamese
        Trained model
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 4, 5]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        'n_tokens': n_tokens,
        'embedding_dim': embedding_dim,
        'kernel_sizes': kernel_sizes,
        'n_filters': n_filters,
        'n_fc_hidden': n_fc_hidden,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'padding_idx': padding_idx,
        'similarity_threshold': similarity_threshold,
        'pooling_strategy': pooling_strategy,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'num_workers': num_workers,
        'accelerator': accelerator,
        'devices': devices,
        'gradient_clip_val': gradient_clip_val,
        'seed': seed,
        'checkpoint_monitor': checkpoint_monitor,
        'checkpoint_mode': checkpoint_mode,
    }

    hyperparameters_file = output_path / 'hyperparameters.yaml'
    with open(hyperparameters_file, 'w') as f:
        yaml.dump(hyperparameters, f, default_flow_style=False, sort_keys=False)
    print(f"Hyperparameters saved to: {hyperparameters_file}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if accelerator == 'gpu' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if accelerator == 'gpu' else False,
    )

    model = CNNSiamese(
        n_tokens=n_tokens,
        embedding_dim=embedding_dim,
        kernel_sizes=kernel_sizes,
        n_filters=n_filters,
        n_fc_hidden=n_fc_hidden,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        padding_idx=padding_idx,
        similarity_threshold=similarity_threshold,
        pooling_strategy=pooling_strategy,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=output_path / 'checkpoints',
        filename=f'{experiment_name or "cnn"}-{{epoch:02d}}-{{val_loss:.4f}}',
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_top_k=1,
        save_last=True,
    )

    metrics_logger = MetricsLogger(output_path)

    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=output_path / 'logs',
        name=experiment_name or 'cnn',
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, metrics_logger],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=gradient_clip_val,
    )

    trainer.fit(model, train_loader, val_loader)

    return model
