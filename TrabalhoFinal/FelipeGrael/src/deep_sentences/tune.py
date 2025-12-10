import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict
import lightning as L
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import os

from .models import RNNSiamese, CNNSiamese
from .train import collate_fn, MetricsLogger

# Disable tokenizers parallelism to avoid warnings with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_trial_model_and_loaders(
    trial: optuna.Trial,
    train_dataset: DatasetDict,
    val_dataset: DatasetDict,
    n_tokens: int,
    padding_idx: int,
    fixed_params: Dict[str, Any],
):
    """
    Create model and data loaders with hyperparameters suggested by Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters
    train_dataset : DatasetDict
        Training dataset
    val_dataset : DatasetDict
        Validation dataset
    n_tokens : int
        Vocabulary size
    padding_idx : int
        Padding token index
    fixed_params : dict
        Fixed hyperparameters not to be tuned

    Returns
    -------
    tuple
        (model, train_loader, val_loader, hyperparameters)
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.7)
    n_hidden = trial.suggest_categorical("n_hidden", [32, 64, 128, 256])
    n_fc_hidden = trial.suggest_categorical("n_fc_hidden", [32, 64, 128, 256])
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.5, 5.0)
    embedding_dim = trial.suggest_categorical("embedding_dim", [50, 100, 200, 300])

    # Get fixed parameters with defaults
    n_layers = fixed_params.get("n_layers", 1)
    rnn_type = fixed_params.get("rnn_type", "lstm")
    bidirectional = fixed_params.get("bidirectional", True)
    similarity_threshold = fixed_params.get("similarity_threshold", 3.0)
    batch_size = fixed_params.get("batch_size", 128)
    num_workers = fixed_params.get("num_workers", 4)
    accelerator = fixed_params.get("accelerator", "auto")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if accelerator == "gpu" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if accelerator == "gpu" else False,
    )

    # Create model
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

    hyperparameters = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "n_hidden": n_hidden,
        "n_fc_hidden": n_fc_hidden,
        "gradient_clip_val": gradient_clip_val,
        "embedding_dim": embedding_dim,
        "n_layers": n_layers,
        "rnn_type": rnn_type,
        "bidirectional": bidirectional,
        "batch_size": batch_size,
        "similarity_threshold": similarity_threshold,
    }

    return model, train_loader, val_loader, hyperparameters


def objective_rnn(
    trial: optuna.Trial,
    train_dataset: DatasetDict,
    val_dataset: DatasetDict,
    n_tokens: int,
    padding_idx: int,
    fixed_params: Dict[str, Any],
    max_epochs: int,
    devices: int,
    accelerator: str,
) -> float:
    """
    Optuna objective function for RNN hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    train_dataset : DatasetDict
        Training dataset
    val_dataset : DatasetDict
        Validation dataset
    n_tokens : int
        Vocabulary size
    padding_idx : int
        Padding token index
    fixed_params : dict
        Fixed hyperparameters
    max_epochs : int
        Maximum training epochs
    devices : int
        Number of devices
    accelerator : str
        Accelerator type

    Returns
    -------
    float
        Validation loss (metric to minimize)
    """
    model, train_loader, val_loader, hyperparameters = create_trial_model_and_loaders(
        trial, train_dataset, val_dataset, n_tokens, padding_idx, fixed_params
    )

    # Create pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    # Create trainer with minimal logging
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[pruning_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        gradient_clip_val=hyperparameters["gradient_clip_val"],
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Return validation loss
    return trainer.callback_metrics["val_loss"].item()


def tune_rnn(
    train_dataset: DatasetDict,
    val_dataset: DatasetDict,
    n_tokens: int,
    padding_idx: int,
    output_dir: str,
    n_trials: int = 50,
    max_epochs: int = 20,
    n_layers: int = 1,
    rnn_type: str = "lstm",
    bidirectional: bool = True,
    similarity_threshold: float = 3.0,
    batch_size: int = 128,
    num_workers: int = 4,
    accelerator: str = "auto",
    devices: int = 1,
    study_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tune RNN hyperparameters using Optuna.

    Parameters
    ----------
    train_dataset : DatasetDict
        Training dataset
    val_dataset : DatasetDict
        Validation dataset
    n_tokens : int
        Vocabulary size
    padding_idx : int
        Padding token index
    output_dir : str
        Directory to save tuning results
    n_trials : int, default=50
        Number of trials to run
    max_epochs : int, default=20
        Maximum epochs per trial
    n_layers : int, default=1
        Number of RNN layers (fixed)
    rnn_type : str, default='lstm'
        RNN type (fixed)
    bidirectional : bool, default=True
        Use bidirectional RNN (fixed)
    similarity_threshold : float, default=3.0
        Similarity threshold (fixed)
    batch_size : int, default=128
        Batch size for training (fixed)
    num_workers : int, default=4
        Number of data loading workers
    accelerator : str, default='auto'
        Accelerator type
    devices : int, default=1
        Number of devices
    study_name : str or None, default=None
        Name for the Optuna study
    timeout : int or None, default=None
        Timeout in seconds for the study

    Returns
    -------
    dict
        Best hyperparameters found
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Fixed parameters
    fixed_params = {
        "n_layers": n_layers,
        "rnn_type": rnn_type,
        "bidirectional": bidirectional,
        "similarity_threshold": similarity_threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "accelerator": accelerator,
    }

    # Create study
    study_db = output_path / "optuna_study.db"
    storage = f"sqlite:///{study_db}"

    study = optuna.create_study(
        study_name=study_name or "rnn_tuning",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )

    print(f"Starting Optuna study with {n_trials} trials...")
    print(f"Study database: {study_db}")

    # Run optimization
    study.optimize(
        lambda trial: objective_rnn(
            trial,
            train_dataset,
            val_dataset,
            n_tokens,
            padding_idx,
            fixed_params,
            max_epochs,
            devices,
            accelerator,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    # Combine with fixed parameters
    all_best_params = {**best_params, **fixed_params}
    all_best_params["n_tokens"] = n_tokens
    all_best_params["padding_idx"] = padding_idx
    all_best_params["best_val_loss"] = best_value

    # Save best parameters
    best_params_file = output_path / "best_hyperparameters.yaml"
    with open(best_params_file, "w") as f:
        yaml.dump(all_best_params, f, default_flow_style=False, sort_keys=False)

    print(f"\nOptimization complete!")
    print(f"Best validation loss: {best_value:.4f}")
    print(f"Best hyperparameters saved to: {best_params_file}")
    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Generate visualizations
    try:
        import optuna.visualization as vis

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_path / "optimization_history.html"))

        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_path / "param_importances.html"))

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(output_path / "parallel_coordinate.html"))

        print(f"\nVisualizations saved to: {output_path}")
    except Exception as e:
        print(f"\nWarning: Could not generate visualizations: {e}")

    return all_best_params


def create_trial_cnn_model_and_loaders(
    trial: optuna.Trial,
    train_dataset: DatasetDict,
    val_dataset: DatasetDict,
    n_tokens: int,
    padding_idx: int,
    fixed_params: Dict[str, Any],
):
    """
    Create CNN model and data loaders with hyperparameters suggested by Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters
    train_dataset : DatasetDict
        Training dataset
    val_dataset : DatasetDict
        Validation dataset
    n_tokens : int
        Vocabulary size
    padding_idx : int
        Padding token index
    fixed_params : dict
        Fixed hyperparameters not to be tuned

    Returns
    -------
    tuple
        (model, train_loader, val_loader, hyperparameters)
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.7)
    n_filters = trial.suggest_categorical("n_filters", [64, 128, 256, 512])
    n_fc_hidden = trial.suggest_categorical("n_fc_hidden", [32, 64, 128, 256])
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 300])
    pooling_strategy = trial.suggest_categorical("pooling_strategy", ["max", "mean", "both"])

    # Suggest kernel sizes configuration
    kernel_config = trial.suggest_categorical("kernel_config", ["small", "medium", "large", "mixed"])
    kernel_sizes_map = {
        "small": [2, 3, 4],
        "medium": [3, 4, 5],
        "large": [4, 5, 6],
        "mixed": [2, 4, 6],
    }
    kernel_sizes = kernel_sizes_map[kernel_config]

    # Get fixed parameters with defaults
    similarity_threshold = fixed_params.get("similarity_threshold", 3.0)
    batch_size = fixed_params.get("batch_size", 128)
    num_workers = fixed_params.get("num_workers", 4)
    accelerator = fixed_params.get("accelerator", "auto")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if accelerator == "gpu" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if accelerator == "gpu" else False,
    )

    # Create model
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

    hyperparameters = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "n_filters": n_filters,
        "n_fc_hidden": n_fc_hidden,
        "embedding_dim": embedding_dim,
        "pooling_strategy": pooling_strategy,
        "kernel_sizes": kernel_sizes,
        "kernel_config": kernel_config,
        "batch_size": batch_size,
        "similarity_threshold": similarity_threshold,
    }

    return model, train_loader, val_loader, hyperparameters


def objective_cnn(
    trial: optuna.Trial,
    train_dataset: DatasetDict,
    val_dataset: DatasetDict,
    n_tokens: int,
    padding_idx: int,
    fixed_params: Dict[str, Any],
    max_epochs: int,
    devices: int,
    accelerator: str,
) -> float:
    """
    Optuna objective function for CNN hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    train_dataset : DatasetDict
        Training dataset
    val_dataset : DatasetDict
        Validation dataset
    n_tokens : int
        Vocabulary size
    padding_idx : int
        Padding token index
    fixed_params : dict
        Fixed hyperparameters
    max_epochs : int
        Maximum training epochs
    devices : int
        Number of devices
    accelerator : str
        Accelerator type

    Returns
    -------
    float
        Validation loss (metric to minimize)
    """
    model, train_loader, val_loader, hyperparameters = create_trial_cnn_model_and_loaders(
        trial, train_dataset, val_dataset, n_tokens, padding_idx, fixed_params
    )

    # Create pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    # Create trainer with minimal logging
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[pruning_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Return validation loss
    return trainer.callback_metrics["val_loss"].item()


def tune_cnn(
    train_dataset: DatasetDict,
    val_dataset: DatasetDict,
    n_tokens: int,
    padding_idx: int,
    output_dir: str,
    n_trials: int = 50,
    max_epochs: int = 20,
    similarity_threshold: float = 3.0,
    batch_size: int = 128,
    num_workers: int = 4,
    accelerator: str = "auto",
    devices: int = 1,
    study_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tune CNN hyperparameters using Optuna.

    Parameters
    ----------
    train_dataset : DatasetDict
        Training dataset
    val_dataset : DatasetDict
        Validation dataset
    n_tokens : int
        Vocabulary size
    padding_idx : int
        Padding token index
    output_dir : str
        Directory to save tuning results
    n_trials : int, default=50
        Number of trials to run
    max_epochs : int, default=20
        Maximum epochs per trial
    similarity_threshold : float, default=3.0
        Similarity threshold (fixed)
    batch_size : int, default=128
        Batch size for training (fixed)
    num_workers : int, default=4
        Number of data loading workers
    accelerator : str, default='auto'
        Accelerator type
    devices : int, default=1
        Number of devices
    study_name : str or None, default=None
        Name for the Optuna study
    timeout : int or None, default=None
        Timeout in seconds for the study

    Returns
    -------
    dict
        Best hyperparameters found
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Fixed parameters
    fixed_params = {
        "similarity_threshold": similarity_threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "accelerator": accelerator,
    }

    # Create study
    study_db = output_path / "optuna_study_cnn.db"
    storage = f"sqlite:///{study_db}"

    study = optuna.create_study(
        study_name=study_name or "cnn_tuning",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )

    print(f"Starting Optuna study for CNN with {n_trials} trials...")
    print(f"Study database: {study_db}")

    # Run optimization
    study.optimize(
        lambda trial: objective_cnn(
            trial,
            train_dataset,
            val_dataset,
            n_tokens,
            padding_idx,
            fixed_params,
            max_epochs,
            devices,
            accelerator,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    # Combine with fixed parameters
    all_best_params = {**best_params, **fixed_params}
    all_best_params["n_tokens"] = n_tokens
    all_best_params["padding_idx"] = padding_idx
    all_best_params["best_val_loss"] = best_value

    # Save best parameters
    best_params_file = output_path / "best_hyperparameters_cnn.yaml"
    with open(best_params_file, "w") as f:
        yaml.dump(all_best_params, f, default_flow_style=False, sort_keys=False)

    print(f"\nOptimization complete!")
    print(f"Best validation loss: {best_value:.4f}")
    print(f"Best hyperparameters saved to: {best_params_file}")
    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Generate visualizations
    try:
        import optuna.visualization as vis

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_path / "optimization_history_cnn.html"))

        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_path / "param_importances_cnn.html"))

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(output_path / "parallel_coordinate_cnn.html"))

        print(f"\nVisualizations saved to: {output_path}")
    except Exception as e:
        print(f"\nWarning: Could not generate visualizations: {e}")

    return all_best_params
