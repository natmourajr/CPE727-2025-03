import typer
from pathlib import Path
from typing import Optional
import lightning as L
from .xval import create_kfold_splits as _create_kfold_splits, get_split_info, load_split
from .dataloaders.stsb import STSBDataset
from .train import train_rnn
from .tune import tune_rnn, tune_cnn
from .tokenizer import train_tokenizer, tokenize_dataset

app = typer.Typer(help="Deep Sentences CLI - Tools for text similarity with Siamese LSTM")


@app.command()
def create_kfold_splits(
    n_folds: int = typer.Option(5, "--n-folds", "-k", help="Number of folds for cross-validation"),
    output: Path = typer.Option("xval_splits.h5", "--output", "-o", help="Output HDF5 file path"),
    no_shuffle: bool = typer.Option(False, "--no-shuffle", help="Disable shuffling before splitting"),
    seed: int = typer.Option(202512, "--seed", "-s", help="Random seed for reproducibility"),
):
    """
    Create k-fold cross-validation splits for STSB dataset and save to HDF5 file.

    Example:
        deep-sentences create-kfold-splits --n-folds 5 --output splits.h5
    """
    typer.echo("Loading STSB dataset...")
    dl = STSBDataset()
    n_samples = len(dl.dataset['dev'])
    typer.echo(f"Dataset loaded: {n_samples} samples in dev set")

    _create_kfold_splits(
        n_samples=n_samples,
        n_folds=n_folds,
        shuffle=not no_shuffle,
        random_state=seed,
        output_path=str(output),
    )


@app.command()
def split_info(
    file_path: Path = typer.Argument(..., help="Path to HDF5 file containing splits"),
):
    """
    Display information about saved k-fold cross-validation splits.

    Example:
        deep-sentences split-info xval_splits.h5
    """
    if not file_path.exists():
        typer.echo(f"Error: File '{file_path}' not found", err=True)
        raise typer.Exit(code=1)

    info = get_split_info(str(file_path))

    typer.echo(f"\n{'='*50}")
    typer.echo(f"Split Information: {file_path}")
    typer.echo(f"{'='*50}")
    typer.echo(f"Total samples:    {info['n_samples']}")
    typer.echo(f"Number of folds:  {info['n_folds']}")
    typer.echo(f"Shuffle:          {info['shuffle']}")
    typer.echo(f"Random state:     {info['random_state']}")
    typer.echo(f"Folds:            {', '.join(info['folds'])}")
    typer.echo(f"{'='*50}\n")


@app.command()
def train(
    split_file: Path = typer.Argument(..., help="Path to HDF5 file containing k-fold splits"),
    fold: int = typer.Argument(..., help="Fold number to train on"),
    experiment_dir: Path = typer.Argument(..., help="Directory to save experiment outputs"),
    embedding_dim: int = typer.Option(300, "--embedding-dim", help="Embedding dimension"),
    n_layers: int = typer.Option(1, "--n-layers", help="Number of RNN layers"),
    n_hidden: int = typer.Option(128, "--n-hidden", help="RNN hidden size"),
    n_fc_hidden: int = typer.Option(128, "--n-fc-hidden", help="Fully connected hidden size"),
    rnn_type: str = typer.Option("lstm", "--rnn-type", help="RNN type: lstm or gru"),
    dropout: float = typer.Option(0.5, "--dropout", help="Dropout rate"),
    learning_rate: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    weight_decay: float = typer.Option(1e-5, "--weight-decay", help="Weight decay"),
    bidirectional: bool = typer.Option(True, "--bidirectional/--unidirectional", help="Use bidirectional RNN"),
    similarity_threshold: float = typer.Option(3.0, "--similarity-threshold", help="Threshold for binary metrics"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    max_epochs: int = typer.Option(50, "--max-epochs", help="Maximum number of epochs"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of data loading workers"),
    accelerator: str = typer.Option("auto", "--accelerator", help="Accelerator type (auto, gpu, cpu)"),
    devices: int = typer.Option(1, "--devices", help="Number of devices"),
    gradient_clip_val: float = typer.Option(1.0, "--gradient-clip", help="Gradient clipping value"),
    seed: int = typer.Option(202512, "--seed", help="Random seed for reproducibility"),
):
    """
    Train Siamese RNN model on a specific fold of the STSB dataset.

    Example:
        deep-sentences train splits.h5 0 experiments/exp1 --max-epochs 100
    """
    # Set random seed for reproducibility
    L.seed_everything(seed, workers=True)
    typer.echo(f"Random seed set to: {seed}")

    if not split_file.exists():
        typer.echo(f"Error: Split file '{split_file}' not found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading split from {split_file}, fold {fold}...")
    train_idx, val_idx = load_split(str(split_file), fold)
    typer.echo(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    typer.echo("Loading STSB dataset...")
    dl = STSBDataset()
    full_dataset = dl.dataset['dev']

    train_dataset = full_dataset.select(train_idx)
    val_dataset = full_dataset.select(val_idx)
    typer.echo("Datasets prepared")

    typer.echo("Training tokenizer on training set...")
    tokenizer = train_tokenizer(train_dataset)
    n_tokens = tokenizer.get_vocab_size()
    padding_idx = tokenizer.token_to_id("<PAD>")
    typer.echo(f"Tokenizer trained. Vocabulary size: {n_tokens}, Padding index: {padding_idx}")

    output_dir = experiment_dir / f"fold_{fold:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    typer.echo(f"Tokenizer saved to: {tokenizer_path}")

    typer.echo("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    val_dataset = tokenize_dataset(val_dataset, tokenizer)
    typer.echo("Datasets tokenized")

    typer.echo(f"Output directory: {output_dir}")

    typer.echo("Starting training...")
    model = train_rnn(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
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
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=gradient_clip_val,
        seed=seed,
        output_dir=str(output_dir),
        experiment_name=f"fold_{fold:02d}",
    )

    typer.echo(f"Training complete! Results saved to {output_dir}")


@app.command()
def tune_rnn_model(
    split_file: Path = typer.Argument(..., help="Path to HDF5 file containing k-fold splits"),
    fold: int = typer.Argument(..., help="Fold number to use for tuning"),
    output_dir: Path = typer.Argument(..., help="Directory to save tuning results"),
    n_trials: int = typer.Option(50, "--n-trials", "-n", help="Number of Optuna trials"),
    max_epochs: int = typer.Option(20, "--max-epochs", help="Maximum epochs per trial"),
    n_layers: int = typer.Option(1, "--n-layers", help="Number of RNN layers (fixed)"),
    rnn_type: str = typer.Option("lstm", "--rnn-type", help="RNN type: lstm or gru (fixed)"),
    bidirectional: bool = typer.Option(True, "--bidirectional/--unidirectional", help="Use bidirectional RNN (fixed)"),
    similarity_threshold: float = typer.Option(3.0, "--similarity-threshold", help="Threshold for binary metrics (fixed)"),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size for training (fixed during tuning)"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of data loading workers"),
    accelerator: str = typer.Option("auto", "--accelerator", help="Accelerator type (auto, gpu, cpu)"),
    devices: int = typer.Option(1, "--devices", help="Number of devices"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Timeout in seconds for the study"),
    seed: int = typer.Option(202512, "--seed", help="Random seed for reproducibility"),
):
    """
    Tune RNN hyperparameters using Optuna on a specific fold.

    This will search for optimal values of:
    - learning_rate, weight_decay, dropout
    - n_hidden, n_fc_hidden, embedding_dim
    - gradient_clip_val

    Fixed parameters: batch_size=128

    Example:
        deep-sentences tune-rnn-model splits.h5 0 experiments/tuning --n-trials 100
    """
    # Set random seed for reproducibility
    L.seed_everything(seed, workers=True)
    typer.echo(f"Random seed set to: {seed}")

    if not split_file.exists():
        typer.echo(f"Error: Split file '{split_file}' not found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading split from {split_file}, fold {fold}...")
    train_idx, val_idx = load_split(str(split_file), fold)
    typer.echo(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    typer.echo("Loading STSB dataset...")
    dl = STSBDataset()
    full_dataset = dl.dataset['dev']

    train_dataset = full_dataset.select(train_idx)
    val_dataset = full_dataset.select(val_idx)
    typer.echo("Datasets prepared")

    typer.echo("Training tokenizer on training set...")
    tokenizer = train_tokenizer(train_dataset)
    n_tokens = tokenizer.get_vocab_size()
    padding_idx = tokenizer.token_to_id("<PAD>")
    typer.echo(f"Tokenizer trained. Vocabulary size: {n_tokens}, Padding index: {padding_idx}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    typer.echo(f"Tokenizer saved to: {tokenizer_path}")

    typer.echo("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    val_dataset = tokenize_dataset(val_dataset, tokenizer)
    typer.echo("Datasets tokenized")

    typer.echo(f"\nStarting hyperparameter tuning with {n_trials} trials...")
    typer.echo(f"Results will be saved to: {output_dir}")

    best_params = tune_rnn(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_tokens=n_tokens,
        padding_idx=padding_idx,
        output_dir=str(output_dir),
        n_trials=n_trials,
        max_epochs=max_epochs,
        n_layers=n_layers,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        accelerator=accelerator,
        devices=devices,
        study_name=f"fold_{fold:02d}_tuning",
        timeout=timeout,
    )

    typer.echo(f"\n{'='*60}")
    typer.echo("RNN Tuning complete!")
    typer.echo(f"Best hyperparameters saved to: {output_dir}/best_hyperparameters.yaml")
    typer.echo(f"Study database: {output_dir}/optuna_study.db")
    typer.echo(f"Visualizations: {output_dir}/*.html")
    typer.echo(f"{'='*60}")


@app.command()
def tune_cnn_model(
    split_file: Path = typer.Argument(..., help="Path to HDF5 file containing k-fold splits"),
    fold: int = typer.Argument(..., help="Fold number to use for tuning"),
    output_dir: Path = typer.Argument(..., help="Directory to save tuning results"),
    n_trials: int = typer.Option(50, "--n-trials", "-n", help="Number of Optuna trials"),
    max_epochs: int = typer.Option(20, "--max-epochs", help="Maximum epochs per trial"),
    similarity_threshold: float = typer.Option(3.0, "--similarity-threshold", help="Threshold for binary metrics (fixed)"),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size for training (fixed during tuning)"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of data loading workers"),
    accelerator: str = typer.Option("auto", "--accelerator", help="Accelerator type (auto, gpu, cpu)"),
    devices: int = typer.Option(1, "--devices", help="Number of devices"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Timeout in seconds for the study"),
    seed: int = typer.Option(202512, "--seed", help="Random seed for reproducibility"),
):
    """
    Tune CNN hyperparameters using Optuna on a specific fold.

    This will search for optimal values of:
    - learning_rate, weight_decay, dropout
    - n_filters, n_fc_hidden, embedding_dim
    - pooling_strategy, kernel_sizes

    Fixed parameters: batch_size=128

    Example:
        deep-sentences tune-cnn-model splits.h5 0 experiments/tuning_cnn --n-trials 100
    """
    # Set random seed for reproducibility
    L.seed_everything(seed, workers=True)
    typer.echo(f"Random seed set to: {seed}")

    if not split_file.exists():
        typer.echo(f"Error: Split file '{split_file}' not found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading split from {split_file}, fold {fold}...")
    train_idx, val_idx = load_split(str(split_file), fold)
    typer.echo(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    typer.echo("Loading STSB dataset...")
    dl = STSBDataset()
    full_dataset = dl.dataset['dev']

    train_dataset = full_dataset.select(train_idx)
    val_dataset = full_dataset.select(val_idx)
    typer.echo("Datasets prepared")

    typer.echo("Training tokenizer on training set...")
    tokenizer = train_tokenizer(train_dataset)
    n_tokens = tokenizer.get_vocab_size()
    padding_idx = tokenizer.token_to_id("<PAD>")
    typer.echo(f"Tokenizer trained. Vocabulary size: {n_tokens}, Padding index: {padding_idx}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    typer.echo(f"Tokenizer saved to: {tokenizer_path}")

    typer.echo("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    val_dataset = tokenize_dataset(val_dataset, tokenizer)
    typer.echo("Datasets tokenized")

    typer.echo(f"\nStarting CNN hyperparameter tuning with {n_trials} trials...")
    typer.echo(f"Results will be saved to: {output_dir}")

    best_params = tune_cnn(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_tokens=n_tokens,
        padding_idx=padding_idx,
        output_dir=str(output_dir),
        n_trials=n_trials,
        max_epochs=max_epochs,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        accelerator=accelerator,
        devices=devices,
        study_name=f"fold_{fold:02d}_cnn_tuning",
        timeout=timeout,
    )

    typer.echo(f"\n{'='*60}")
    typer.echo("CNN Tuning complete!")
    typer.echo(f"Best hyperparameters saved to: {output_dir}/best_hyperparameters_cnn.yaml")
    typer.echo(f"Study database: {output_dir}/optuna_study_cnn.db")
    typer.echo(f"Visualizations: {output_dir}/*_cnn.html")
    typer.echo(f"{'='*60}")


def main() -> None:
    app()
