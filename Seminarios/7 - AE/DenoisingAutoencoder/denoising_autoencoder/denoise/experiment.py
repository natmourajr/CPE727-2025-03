"""Denoising experiment: Train DAE to denoise MNIST images with cross-validation."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import mlflow
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from mnist_loader.loader import MNISTDataset
from mnist_loader.noise_mnist.loader import NoisyMNISTDataset
from denoising_autoencoder.models.denoising_autoencoder import ConvolutionalDenoisingAutoencoder
from denoising_autoencoder.utils import (
    get_device, calculate_mse, calculate_ssim, save_metrics_as_artifact
)


# Constants
DATA_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "Data"
print(f"Data path: {DATA_PATH}")


def load_cv_splits(csv_path, fold):
    """Load train/val indices for a specific cross-validation fold.

    The CSV format stores which folds each sample is used for TRAINING.
    For a 5-fold CV, each sample appears 4 times (the 4 folds where it's in training).
    The missing fold is where that sample is used for validation.
    """
    df = pd.read_csv(csv_path)

    # Get all unique indices
    all_indices = df['index'].unique()

    # Train: samples where split == fold (samples marked for training in this fold)
    train_indices = df[df['split'] == fold]['index'].unique().tolist()

    # Val: samples NOT in the training set (the missing fold for each sample)
    val_indices = [idx for idx in all_indices if idx not in train_indices]

    return train_indices, val_indices


def create_dataloaders(train_indices, val_indices, config, test=False):
    """Create train, val, and test dataloaders."""
    # Load noisy training dataset
    full_train_dataset = NoisyMNISTDataset(
        download_path=DATA_PATH / "NoisyMnist",
        train=True,
        noise_level=config['noise_level'],
        noise_seed=config['noise_seed']
    )

    # Create subsets using CV splits
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_loader=None
    # Create dataloaders
    if train_indices:

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2
        )

    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )

    # Test loader (only load once if requested)
    test_loader = None
    if test:
        test_dataset = NoisyMNISTDataset(
            download_path=DATA_PATH / "NoisyMnist",
            train=False,
            noise_level=config['noise_level'],
            noise_seed=config['noise_seed'] + 1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )

    return train_loader, val_loader, test_loader


def build_model(config, device):
    """Build and return autoencoder model."""
    model = ConvolutionalDenoisingAutoencoder(
        in_channels=1,
        latent_dim=config['latent_dim'],
        image_size=28
    ).to(device)
    return model


def train_model(model, train_loader, val_loader, config, device):
    """Train the denoising autoencoder and return training history."""
    print(f"\nStarting training with:")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Epochs: {config['epochs']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Device: {device}\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )

    train_losses = []
    val_losses = []
    val_mses = []
    val_ssims = []

    # Early stopping configuration
    patience = config.get('early_stopping_patience', 10)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        epoch_loss = 0.0

        print(f"Epoch {epoch+1}/{config['epochs']} - Training...", end='', flush=True)
        for noisy_x, clean_x, _ in tqdm(train_loader, desc=f"  Training", leave=False, ncols=80):
            noisy_x = noisy_x.unsqueeze(1).to(device)
            clean_x = clean_x.unsqueeze(1).to(device)

            optimizer.zero_grad()
            reconstructed = model(noisy_x)
            loss = criterion(reconstructed, clean_x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f" Train Loss: {avg_train_loss:.4f}", end='', flush=True)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_ssim = 0.0

        print(f" | Validating...", end='', flush=True)
        with torch.no_grad():
            for noisy_x, clean_x, _ in tqdm(val_loader, desc=f"  Validation", leave=False, ncols=80):
                noisy_x = noisy_x.unsqueeze(1).to(device)
                clean_x = clean_x.unsqueeze(1).to(device)
                reconstructed = model(noisy_x)

                val_loss += criterion(reconstructed, clean_x).item()
                val_mse += calculate_mse(reconstructed, clean_x)
                val_ssim += calculate_ssim(reconstructed, clean_x)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)

        val_losses.append(avg_val_loss)
        val_mses.append(avg_val_mse)
        val_ssims.append(avg_val_ssim)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f" Val Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, SSIM: {avg_val_ssim:.4f} *** Best Model ***")
        else:
            early_stopping_counter += 1
            print(f" Val Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, SSIM: {avg_val_ssim:.4f} (No improvement for {early_stopping_counter} epochs)")

            if early_stopping_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs (patience={patience})")
                print(f"Best model was at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model from epoch {best_epoch}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_mses': val_mses,
        'val_ssims': val_ssims,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }


def evaluate_model(model, test_loader, device, save_visualization=True):
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.MSELoss()

    test_loss = 0.0
    test_mse = 0.0
    test_ssim = 0.0

    # Collect samples for visualization
    vis_samples = {'noisy': [], 'clean': [], 'reconstructed': []}
    num_vis_samples = 16

    with torch.no_grad():
        for noisy_x, clean_x, _ in tqdm(test_loader, desc="Testing", leave=False):
            noisy_x = noisy_x.unsqueeze(1).to(device)
            clean_x = clean_x.unsqueeze(1).to(device)
            reconstructed = model(noisy_x)

            test_loss += criterion(reconstructed, clean_x).item()
            test_mse += calculate_mse(reconstructed, clean_x)
            test_ssim += calculate_ssim(reconstructed, clean_x)

            # Collect samples for visualization
            if save_visualization and len(vis_samples['noisy']) < num_vis_samples:
                batch_size = noisy_x.size(0)
                remaining = num_vis_samples - len(vis_samples['noisy'])
                samples_to_add = min(batch_size, remaining)

                vis_samples['noisy'].extend(noisy_x[:samples_to_add].cpu())
                vis_samples['clean'].extend(clean_x[:samples_to_add].cpu())
                vis_samples['reconstructed'].extend(reconstructed[:samples_to_add].cpu())

    test_loss = test_loss / len(test_loader)
    test_mse = test_mse / len(test_loader)
    test_ssim = test_ssim / len(test_loader)

    # Create visualization if requested
    viz_path = None
    if save_visualization and len(vis_samples['noisy']) > 0:
        viz_path = create_visualization_grid(vis_samples)

    return test_loss, test_mse, test_ssim, viz_path


def create_visualization_grid(samples):
    """Create and save grid of original, noisy, and reconstructed images with metrics."""
    n_samples = len(samples['noisy'])
    n_cols = min(8, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(n_cols * 2, n_rows * 6))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows * 3, n_cols)

    for idx in range(n_samples):
        col = idx % n_cols
        row_base = (idx // n_cols) * 3

        # Clean image
        axes[row_base, col].imshow(samples['clean'][idx].squeeze(), cmap='gray')
        axes[row_base, col].axis('off')
        if col == 0:
            axes[row_base, col].set_title('Clean', fontsize=10)

        # Noisy image
        axes[row_base + 1, col].imshow(samples['noisy'][idx].squeeze(), cmap='gray')
        axes[row_base + 1, col].axis('off')
        if col == 0:
            axes[row_base + 1, col].set_title('Noisy', fontsize=10)

        # Reconstructed image
        axes[row_base + 2, col].imshow(samples['reconstructed'][idx].squeeze(), cmap='gray')
        axes[row_base + 2, col].axis('off')

        # Calculate metrics for this sample
        clean = samples['clean'][idx].unsqueeze(0)
        reconstructed = samples['reconstructed'][idx].unsqueeze(0)
        mse = calculate_mse(reconstructed, clean)
        ssim = calculate_ssim(reconstructed, clean)

        # Add metrics as text below the image
        metrics_text = f'MSE: {mse:.4f}\nSSIM: {ssim:.4f}'
        axes[row_base + 2, col].text(0.5, -0.1, metrics_text,
                                      transform=axes[row_base + 2, col].transAxes,
                                      ha='center', va='top', fontsize=8)

        if col == 0:
            axes[row_base + 2, col].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path = 'denoising_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def train_single_fold(config, fold, cv_splits_path, device, test_loader=None):
    """Train and evaluate a single fold of cross-validation."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")

    # Load CV splits
    print(f"Loading CV splits for fold {fold}...")
    train_indices, val_indices = load_cv_splits(cv_splits_path, fold)
    print(f"  - Training samples: {len(train_indices)}")
    print(f"  - Validation samples: {len(val_indices)}")

    print(f"Creating data loaders...")
    train_loader, val_loader, _ = create_dataloaders(train_indices, val_indices, config, test=False)

    # Build model
    print(f"Building model (latent_dim={config['latent_dim']})...")
    model = build_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Train model
    history = train_model(model, train_loader, val_loader, config, device)
    print(f"\nTraining completed!")
    print(f"  - Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"  - Final val loss: {history['val_losses'][-1]:.4f}")
    print(f"  - Final val SSIM: {history['val_ssims'][-1]:.4f}")

    # Evaluate on test set if provided
    test_metrics = None
    viz_path = None
    if test_loader is not None:
        print(f"\nEvaluating on test set...")
        test_loss, test_mse, test_ssim, viz_path = evaluate_model(
            model, test_loader, device, save_visualization=(fold == 0)
        )
        test_metrics = {
            'test_loss': float(test_loss),
            'test_mse': float(test_mse),
            'test_ssim': float(test_ssim)
        }
        print(f"Fold {fold} Test - Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, SSIM: {test_ssim:.4f}")

    return model, history, test_metrics, viz_path


def run_grid_search_with_cv(param_file, cv_splits_path, experiment_name="denoising_cv"):
    """Run grid search with cross-validation from YAML parameter file.

    For each hyperparameter combination, trains n_folds models and logs all to MLflow.
    """
    with open(param_file, 'r') as f:
        grid_config = yaml.safe_load(f)

    base_config = grid_config['base']
    grid_params = grid_config['grid']

    # Get device
    device = get_device()

    # Determine number of folds from CV splits file
    print(f"\nLoading cross-validation configuration...")
    df = pd.read_csv(cv_splits_path)
    n_folds = df['split'].nunique()
    total_samples = df['index'].nunique()
    print(f"  - Cross-validation with {n_folds} folds")
    print(f"  - Total training samples: {total_samples:,}")
    print(f"  - Device: {device}")

    # Generate all parameter combinations
    keys = list(grid_params.keys())
    values = list(grid_params.values())
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    print(f"\nGrid search configuration:")
    for key, vals in zip(keys, values):
        print(f"  - {key}: {vals}")
    print(f"  - Total combinations: {total_combinations}")
    print(f"  - Total models to train: {total_combinations * n_folds}\n")

    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment: {experiment_name}")

    all_results = []

    # Iterate through each hyperparameter combination
    for param_idx, combination in enumerate(itertools.product(*values)):
        config = base_config.copy()
        for key, value in zip(keys, combination):
            config[key] = value

        print(f"\n{'='*80}")
        print(f"Grid Search {param_idx + 1}/{total_combinations}")
        print(f"Config: {config}")
        print(f"{'='*80}")

        # Create test loader once per configuration
        print(f"Creating test data loader...")
        _, _, test_loader = create_dataloaders([], [], config, test=True)
        if test_loader:
            print(f"  - Test batches: {len(test_loader)}")

        # Parent run for this hyperparameter combination
        with mlflow.start_run(run_name=f"grid_{param_idx+1}") as parent_run:
            # Log hyperparameters
            mlflow.log_params(config)
            mlflow.log_param("device", str(device))
            mlflow.log_param("n_folds", n_folds)

            fold_results = []

            # Train each fold
            print(f"\nTraining {n_folds} folds for this configuration...")
            for fold in range(n_folds):
                # Nested run for each fold
                print(f"\n[Fold {fold+1}/{n_folds}]")
                with mlflow.start_run(run_name=f"fold_{fold}", nested=True) as fold_run:
                    mlflow.log_params(config)
                    mlflow.log_param("fold", fold)
                    mlflow.log_param("parent_run_id", parent_run.info.run_id)

                    # Train single fold
                    model, history, test_metrics, viz_path = train_single_fold(
                        config, fold, cv_splits_path, device, test_loader
                    )

                    # Log training history
                    artifact_path = f"fold_{fold}_training_metrics.json"
                    save_metrics_as_artifact(history, artifact_path)
                    mlflow.log_artifact(artifact_path)

                    # Log validation metrics (final epoch)
                    mlflow.log_metric("final_val_loss", history['val_losses'][-1])
                    mlflow.log_metric("final_val_mse", history['val_mses'][-1])
                    mlflow.log_metric("final_val_ssim", float(history['val_ssims'][-1]))

                    # Log test metrics if available
                    if test_metrics:
                        mlflow.log_metric("test_loss", test_metrics['test_loss'])
                        mlflow.log_metric("test_mse", test_metrics['test_mse'])
                        mlflow.log_metric("test_ssim", test_metrics['test_ssim'])
                        fold_results.append(test_metrics)

                    # Log visualization for first fold
                    if viz_path:
                        mlflow.log_artifact(viz_path)

                    # Save model
                    model_path = f"model_fold_{fold}.pth"
                    torch.save(model.state_dict(), model_path)
                    mlflow.log_artifact(model_path)

                    print(f"Fold {fold} completed - Run ID: {fold_run.info.run_id}")

            # Aggregate results across folds
            if fold_results:
                avg_test_loss = np.mean([r['test_loss'] for r in fold_results])
                avg_test_mse = np.mean([r['test_mse'] for r in fold_results])
                avg_test_ssim = np.mean([r['test_ssim'] for r in fold_results])

                std_test_loss = np.std([r['test_loss'] for r in fold_results])
                std_test_mse = np.std([r['test_mse'] for r in fold_results])
                std_test_ssim = np.std([r['test_ssim'] for r in fold_results])

                # Log aggregated metrics to parent run
                mlflow.log_metric("avg_test_loss", float(avg_test_loss))
                mlflow.log_metric("avg_test_mse", float(avg_test_mse))
                mlflow.log_metric("avg_test_ssim", float(avg_test_ssim))
                mlflow.log_metric("std_test_loss", float(std_test_loss))
                mlflow.log_metric("std_test_mse", float(std_test_mse))
                mlflow.log_metric("std_test_ssim", float(std_test_ssim))

                print(f"\n{'='*60}")
                print(f"Grid {param_idx + 1} Summary:")
                print(f"Avg Test Loss: {avg_test_loss:.4f} ± {std_test_loss:.4f}")
                print(f"Avg Test MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}")
                print(f"Avg Test SSIM: {avg_test_ssim:.4f} ± {std_test_ssim:.4f}")
                print(f"{'='*60}")

                all_results.append({
                    'config': config,
                    'avg_test_loss': avg_test_loss,
                    'avg_test_mse': avg_test_mse,
                    'avg_test_ssim': avg_test_ssim,
                    'std_test_loss': std_test_loss,
                    'std_test_mse': std_test_mse,
                    'std_test_ssim': std_test_ssim,
                    'fold_results': fold_results
                })

    return all_results
