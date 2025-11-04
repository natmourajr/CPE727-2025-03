import sys
import os
import json
from typing import Dict, Any, List

# Add paths for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../modules'))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import schedulers factory
from src.modules.optimization.scheduler import Scheduler

# Import breast cancer loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../dataloaders/BreastCancerLoader'))
from breast_cancer_loader import BreastCancerDataset

# Import preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../modules/preprocessing/BreastCancer'))
from breast_cancer_preprocessing import encode_dataset

# Import SimpleMLP model
from simple_mlp import SimpleMLP

# Import logger
from logger import ExperimentLogger


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _plot_and_save(train_losses: List[float], save_path: str, title: str = "Train Loss"):
    if len(train_losses) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def _plot_lr(lrs: List[List[float]], save_path: str, title: str = "Learning Rate"):
    if len(lrs) == 0:
        return
    plt.figure(figsize=(8, 5))
    lrs_array = torch.tensor(lrs)  # shape: (epochs, param_groups)
    for i in range(lrs_array.shape[1]):
        plt.plot(range(1, lrs_array.shape[0] + 1), lrs_array[:, i].numpy(), marker='o', linewidth=1, label=f'param_group {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_breast_cancer_mlp(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    hidden_size=64,
    feature_strategy='onehot',
    target_strategy='binary',
    handle_missing='drop',
    device='cpu',
    scheduler_name: str = 'CosineAnnealingLR',
    scheduler_params: str = '{}'
):
    """Train SimpleMLP on Breast Cancer dataset with preprocessing (Adam + flexible LR schedulers)"""
    # Initialize logger
    repo_root = os.path.join(os.path.dirname(__file__), '../../../..')
    results_dir = os.path.abspath(os.path.join(repo_root, 'results'))
    logger = ExperimentLogger('breast_cancer_mlp', results_dir=results_dir)

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.log(f'Using device: {device}')

    # Load raw datasets
    logger.log('Loading Breast Cancer dataset...')
    train_dataset = BreastCancerDataset(split='train', train_ratio=0.8, random_state=42)
    test_dataset = BreastCancerDataset(split='test', train_ratio=0.8, random_state=42)

    # Convert to pandas DataFrames
    X_train = train_dataset.X
    y_train = train_dataset.y
    X_test = test_dataset.X
    y_test = test_dataset.y

    logger.log(f'Raw train samples: {len(X_train)}')
    logger.log(f'Raw test samples: {len(X_test)}')

    # Apply preprocessing - fit on train, transform on test
    logger.log(f'Applying preprocessing (feature={feature_strategy}, target={target_strategy}, missing={handle_missing})...')
    X_train_encoded, y_train_encoded, encoder = encode_dataset(
        X_train, y_train,
        feature_strategy=feature_strategy,
        target_strategy=target_strategy,
        handle_missing=handle_missing,
        encoder=None  # Fit encoder on train
    )
    X_test_encoded, y_test_encoded, _ = encode_dataset(
        X_test, y_test,
        feature_strategy=feature_strategy,
        target_strategy=target_strategy,
        handle_missing=handle_missing,
        encoder=encoder  # Reuse fitted encoder from train
    )

    logger.log(f'Encoded train samples: {X_train_encoded.shape[0]}')
    logger.log(f'Encoded test samples: {X_test_encoded.shape[0]}')
    logger.log(f'Feature dimension: {X_train_encoded.shape[1]}')
    logger.log('')

    # Create TensorDatasets for DataLoader
    train_dataset_encoded = torch.utils.data.TensorDataset(X_train_encoded, y_train_encoded)
    test_dataset_encoded = torch.utils.data.TensorDataset(X_test_encoded, y_test_encoded)

    train_loader = torch.utils.data.DataLoader(train_dataset_encoded, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset_encoded, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = X_train_encoded.shape[1]
    output_size = 1  # Binary classification
    model = SimpleMLP(input_size, hidden_size, output_size).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'Model: SimpleMLP')
    logger.log(f'Total parameters: {total_params:,}')
    logger.log('')

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Parse scheduler params JSON string
    try:
        sched_params = json.loads(scheduler_params) if isinstance(scheduler_params, str) else dict(scheduler_params)
    except Exception:
        logger.log('Invalid scheduler_params JSON string; falling back to {}')
        sched_params = {}

    # Build scheduler using SchedulerFactory
    steps_per_epoch = len(train_loader)
    factory = Scheduler(optimizer, num_epochs=epochs, steps_per_epoch=steps_per_epoch)
    scheduler_obj = factory.create(scheduler_name, sched_params)

    # Record configuration
    logger.log_config({
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'hidden_size': hidden_size,
        'feature_strategy': feature_strategy,
        'target_strategy': target_strategy,
        'handle_missing': handle_missing,
        'optimizer': 'Adam',
        'loss': 'BCEWithLogitsLoss',
        'scheduler': scheduler_name,
        'scheduler_params': sched_params
    })

    # Decide stepping granularity for known batch-based schedulers
    batch_step_schedulers = ('CyclicLR', 'OneCycleLR')
    scheduler_steps_per_batch = scheduler_name in batch_step_schedulers

    # Prepare to record training loss per epoch
    train_losses_per_epoch: List[float] = []
    lrs_per_epoch: List[List[float]] = []

    # Ensure results subdir exists
    experiment_results_dir = os.path.join(results_dir, 'scheduler')
    _ensure_dir(experiment_results_dir)

    # Training loop
    logger.log('=== Training ===')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Ensure y_batch is correct shape for BCE loss
            if len(y_batch.shape) == 1:
                y_batch = y_batch.unsqueeze(1).float()

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # If scheduler requires step per batch (OneCycleLR / CyclicLR), call it here
            if scheduler_steps_per_batch:
                try:
                    scheduler_obj.step()
                except TypeError:
                    scheduler_obj.step()

            train_loss += loss.item()

        # Average loss
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else train_loss
        train_losses_per_epoch.append(avg_train_loss)

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                if len(y_batch.shape) == 1:
                    y_batch = y_batch.unsqueeze(1).float()

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

        avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else test_loss
        accuracy = 100 * correct / total if total > 0 else 0.0

        # Scheduler stepping for epoch-based schedulers
        if not scheduler_steps_per_batch:
            if isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_obj.step(avg_test_loss)
            else:
                try:
                    scheduler_obj.step()
                except TypeError:
                    scheduler_obj.step()

        # Log metrics
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        lrs_per_epoch.append(current_lrs.copy())

        if len(current_lrs) == 1:
            lr_for_logging = float(current_lrs[0])
        else:
            lr_for_logging = ','.join([f'{lr:.6g}' for lr in current_lrs])


        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log_metrics(
                epoch=f'{epoch+1}/{epochs}',
                metrics_dict={
                    'Train Loss': avg_train_loss,
                    'Test Loss': avg_test_loss,
                    'Accuracy (%)': accuracy,
                    'LR': lr_for_logging
                }
            )

    logger.log('')
    logger.log('=== Training Complete ===')
    logger.log(f'Final Train Loss: {avg_train_loss:.4f}')
    logger.log(f'Final Test Loss: {avg_test_loss:.4f}')
    logger.log(f'Final Test Accuracy: {accuracy:.2f}%')

    # Save train loss plot
    plot_filename = f'train_loss_per_epoch_{scheduler_name}.png'
    plot_path = os.path.join(experiment_results_dir, plot_filename)
    _plot_and_save(train_losses_per_epoch, plot_path, title='Train Loss per Epoch')
    logger.log(f'Train loss plot saved to: {plot_path}')

    # Save learning rate plot
    lr_plot_filename = f'learning_rate_per_epoch_{scheduler_name}.png'
    lr_plot_path = os.path.join(experiment_results_dir, lr_plot_filename)
    _plot_lr(lrs_per_epoch, lr_plot_path, title='Learning Rate per Epoch')
    logger.log(f'Learning rate plot saved to: {lr_plot_path}')

    # Close logger
    logger.close()


if __name__ == '__main__':
    train_breast_cancer_mlp()
