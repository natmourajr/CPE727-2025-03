"""CLI for Denoising Autoencoder experiments."""
from pathlib import Path
import typer
from typing import Optional
from mnist_loader.loader import MNISTDataset
from mnist_loader.noise_mnist.loader import NoisyMNISTDataset
from mnist_loader import generate_cv_splits, export_cv_splits_to_csv
from denoising_autoencoder.denoise.experiment import run_grid_search_with_cv

app = typer.Typer(help="Denoising Autoencoder Experiments")


@app.command()
def generate_cross_validation_splits(
    output_dir: str = typer.Option(
        "../../../Data/autoencoder/splits/",
        "--output-dir", "-o",
        help="Directory to save the generated splits"
    ),
    n_splits: int = typer.Option(
        5,
        "--n-splits", "-n",
        help="Number of cross-validation splits to generate"
    ),
    random_seed: Optional[int] = typer.Option(
        42,
        "--random-seed", "-r",
        help="Random seed for reproducibility"
    ),
    noise_level: float = typer.Option(
        0.3,
        "--noise-level",
        help="Noise level for Noisy MNIST dataset"
    )
):
    """Generate cross-validation splits for MNIST and Noisy MNIST datasets."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_root = Path("../../../Data")

    typer.echo(f"Generating {n_splits} cross-validation splits...")
    typer.echo(f"Random seed: {random_seed}")

    # Generate splits for regular MNIST
    typer.echo("\n1. Processing MNIST dataset...")
    mnist_dataset = MNISTDataset(download_path=str(data_root / "MNIST"), train=True)
    mnist_splits = generate_cv_splits(mnist_dataset, n_splits=n_splits, random_state=random_seed)

    # Export to CSV
    mnist_csv_path = output_path / "mnist_cv_splits.csv"
    mnist_summary_path = output_path / "mnist_cv_summary.csv"

    typer.echo(f"   Exporting splits to {mnist_csv_path}...")
    export_cv_splits_to_csv(mnist_dataset, mnist_splits, mnist_csv_path)

    # Generate splits for Noisy MNIST
    typer.echo(f"\n2. Processing Noisy MNIST dataset (noise_level={noise_level})...")
    noisy_mnist_dataset = NoisyMNISTDataset(
        download_path=str(data_root / "NoisyMnist"),
        train=True,
        noise_level=noise_level,
        noise_seed=random_seed
    )
    noisy_splits = generate_cv_splits(noisy_mnist_dataset, n_splits=n_splits, random_state=random_seed)

    # Export to CSV
    noisy_csv_path = output_path / f"noisy_mnist_cv_splits_noise{noise_level}.csv"
    noisy_summary_path = output_path / f"noisy_mnist_cv_summary_noise{noise_level}.csv"

    typer.echo(f"   Exporting splits to {noisy_csv_path}...")
    export_cv_splits_to_csv(noisy_mnist_dataset, noisy_splits, noisy_csv_path)

    typer.echo(f"\n✓ Splits saved to: {output_dir}")
    typer.echo(f"  - {mnist_csv_path.name}")
    typer.echo(f"  - {noisy_csv_path.name}")
    typer.echo(f"  - {noisy_summary_path.name}")


@app.command()
def denoise(
    params: str = typer.Option(
        "../../../Data/autoencoder/params/denoising_params.yaml",
        "--params", "-p",
        help="Path to YAML parameter file"
    ),
    cv_splits: str = typer.Option(
        "../../../Data/autoencoder/splits/noisy_mnist_cv_splits_noise0.3.csv",
        "--cv-splits", "-cv",
        help="Path to cross-validation splits CSV file"
    ),
    experiment_name: str = typer.Option(
        "denoising_cv",
        "--experiment-name", "-e",
        help="MLflow experiment name"
    )
):
    """Run denoising experiment with cross-validation to reconstruct clean images."""
    typer.echo(f"Running denoising experiment with params: {params}")
    typer.echo(f"Using CV splits: {cv_splits}")
    results = run_grid_search_with_cv(
        param_file=params,
        cv_splits_path=cv_splits,
        experiment_name=experiment_name
    )
    typer.echo("\nExperiment completed successfully!")
    typer.echo(f"Total configurations tested: {len(results)}")

    # Show best result
    if results:
        best_idx = max(range(len(results)), key=lambda i: results[i]['avg_test_ssim'])
        typer.echo(f"\nBest Configuration (by SSIM):")
        typer.echo(f"  Config: {results[best_idx]['config']}")
        typer.echo(f"  Avg SSIM: {results[best_idx]['avg_test_ssim']:.4f} ± {results[best_idx]['std_test_ssim']:.4f}")


@app.command()
def info():
    """Display information about available experiments."""
    info_text = """
    Denoising Autoencoder Experiments
    ==================================

    Available commands:

    1. generate-cross-validation-splits: Generate CV splits for datasets
       - Creates stratified K-fold splits for MNIST and Noisy MNIST
       - Exports splits to CSV with fold information
       - Generates summary statistics per fold
       - Configurable noise level and number of splits

    2. classify: Train a classifier on DAE latent representations
       - Uses noisy MNIST for training
       - Extracts latent features using trained DAE
       - Trains MLP classifier on latent space
       - Evaluates classification accuracy

    3. denoise: Train DAE to reconstruct clean images from noisy ones (with CV)
       - Performs grid search with K-fold cross-validation
       - Each hyperparameter combination trains N models (one per fold)
       - All models are logged to MLflow with nested runs
       - Evaluates with MSE and SSIM metrics
       - Generates visualization grids
       - Reports mean ± std across folds

    Usage:
      denoising-autoencoder generate-cross-validation-splits --n-splits 5
      denoising-autoencoder classify --params path/to/params.yaml
      denoising-autoencoder denoise --params path/to/params.yaml --cv-splits path/to/splits.csv
    """
    typer.echo(info_text)


if __name__ == "__main__":
    app()