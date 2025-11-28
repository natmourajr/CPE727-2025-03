import typer
from pathlib import Path
from typing import Optional, List

from .download_dataset import download_rcv1
from .train import train as train_model
from .evaluate import encode_dataset, encode_with_reconstruction


app = typer.Typer()


@app.command()
def download_dataset(
    output_dir: str = typer.Argument(
        ...,
        help="Directory where the dataset will be saved"
    ),
    n_features: Optional[int] = typer.Option(
        None,
        "--n-features",
        "-n",
        help="Number of most frequent features to keep (default: all features)"
    )
):
    """
    Download the Reuters RCV1 dataset and save it to HDF5 format.
    """
    output_path = Path(output_dir)
    download_rcv1(output_path, n_features=n_features)


@app.command()
def train(
    dataset_path: str = typer.Argument(
        ...,
        help="Path to the RCV1 HDF5 dataset file"
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directory where outputs (checkpoints, metrics) will be saved"
    ),
    dimensions: str = typer.Argument(
        ...,
        help="Comma-separated list of layer dimensions (e.g., '2000,500,125,2')"
    ),
    mlflow_tracking_uri: Optional[str] = typer.Option(
        None,
        "--mlflow-uri",
        help="MLflow tracking URI or directory (default: ./mlruns)"
    ),
    activation: str = typer.Option(
        "relu",
        "--activation",
        "-a",
        help="Activation function for intermediate layers"
    ),
    latent_activation: str = typer.Option(
        "linear",
        "--latent-activation",
        help="Activation function for latent space"
    ),
    l1_alpha: Optional[float] = typer.Option(
        None,
        "--l1-alpha",
        help="L1 regularization coefficient for latent activations"
    ),
    learning_rate: float = typer.Option(
        3e-4,
        "--learning-rate",
        "--lr",
        help="Learning rate for AdamW optimizer"
    ),
    weight_decay: float = typer.Option(
        0.01,
        "--weight-decay",
        "--wd",
        help="Weight decay for AdamW optimizer"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for training"
    ),
    num_epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Number of training epochs"
    ),
    val_split: float = typer.Option(
        0.2,
        "--val-split",
        help="Fraction of data to use for validation (0.0 to 1.0)"
    ),
    num_workers: int = typer.Option(
        0,
        "--num-workers",
        help="Number of DataLoader workers (use 0 for HDF5)"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility"
    ),
    precision: str = typer.Option(
        "32-true",
        "--precision",
        help="Training precision: '32-true' (float32), 'bf16-mixed' (bfloat16), '16-mixed' (float16). Use 'bf16-mixed' for RTX 4070."
    ),
):
    """
    Train the Autoencoder model on RCV1 dataset.

    Example:
        autoencoder_nlp train data/rcv1_dataset.h5 experiments/run_001 2000,500,125,2
        autoencoder_nlp train data/rcv1_dataset.h5 experiments/run_001 2000,500,125,2 --precision bf16-mixed
    """
    # Parse dimensions string into list of integers
    try:
        dims = [int(d.strip()) for d in dimensions.split(",")]
    except ValueError:
        typer.echo("Error: dimensions must be comma-separated integers (e.g., '2000,500,125,2')", err=True)
        raise typer.Exit(1)

    # Convert paths
    dataset_path_obj = Path(dataset_path)
    output_dir_obj = Path(output_dir)

    # Call train function
    train_model(
        dataset_path=dataset_path_obj,
        output_dir=output_dir_obj,
        dimensions=dims,
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
    )


@app.command()
def evaluate(
    dataset_path: str = typer.Argument(
        ...,
        help="Path to the RCV1 HDF5 dataset file"
    ),
    checkpoint_path: str = typer.Argument(
        ...,
        help="Path to the trained model checkpoint (.ckpt file)"
    ),
    output_path: str = typer.Argument(
        ...,
        help="Path where encoded results will be saved (.h5 file)"
    ),
    batch_size: int = typer.Option(
        256,
        "--batch-size",
        "-b",
        help="Batch size for encoding"
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)"
    ),
    with_reconstruction: bool = typer.Option(
        False,
        "--with-reconstruction",
        "-r",
        help="Also compute reconstruction errors"
    ),
):
    """
    Encode the RCV1 dataset to latent space using a trained autoencoder.

    Example:
        autoencoder_nlp evaluate data/rcv1_dataset.h5 outputs/checkpoints/best.ckpt outputs/latent.h5
        autoencoder_nlp evaluate data/rcv1_dataset.h5 outputs/checkpoints/best.ckpt outputs/latent.h5 --with-reconstruction
    """
    # Convert paths
    dataset_path_obj = Path(dataset_path)
    checkpoint_path_obj = Path(checkpoint_path)
    output_path_obj = Path(output_path)

    # Validate inputs
    if not dataset_path_obj.exists():
        typer.echo(f"Error: Dataset not found: {dataset_path}", err=True)
        raise typer.Exit(1)
    if not checkpoint_path_obj.exists():
        typer.echo(f"Error: Checkpoint not found: {checkpoint_path}", err=True)
        raise typer.Exit(1)

    # Run encoding
    if with_reconstruction:
        encode_with_reconstruction(
            dataset_path=dataset_path_obj,
            checkpoint_path=checkpoint_path_obj,
            output_path=output_path_obj,
            batch_size=batch_size,
            device=device,
        )
    else:
        encode_dataset(
            dataset_path=dataset_path_obj,
            checkpoint_path=checkpoint_path_obj,
            output_path=output_path_obj,
            batch_size=batch_size,
            device=device,
        )


if __name__ == '__main__':
    app()
