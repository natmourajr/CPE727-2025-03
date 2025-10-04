import sys
import os

# Add paths for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../modules'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import wine quality loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../dataloaders/WineQualityLoader'))
from wine_quality_loader import WineQualityDataset

# Import SimpleMLP model
from simple_mlp import SimpleMLP

# Import L1 regularizer
from regularization import L1Regularizer

# Import logger
from logger import ExperimentLogger


def train_wine_mlp(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    hidden_size=64,
    lambda_l1=0.001,
    device='cpu'
):
    """Train SimpleMLP on Wine Quality dataset with L1 regularization

    Args:
        epochs: Number of training epochs (default: 50)
        batch_size: Batch size for training (default: 32)
        learning_rate: Learning rate for optimizer (default: 0.001)
        hidden_size: Size of hidden layer (default: 64)
        lambda_l1: L1 regularization strength (default: 0.001)
        device: Device to train on ('cpu' or 'cuda')
    """
    # Initialize logger
    repo_root = os.path.join(os.path.dirname(__file__), '../../../..')
    results_dir = os.path.abspath(os.path.join(repo_root, 'results'))
    logger = ExperimentLogger('wine_mlp_l1', results_dir=results_dir)

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.log(f'Using device: {device}')

    # Load datasets
    logger.log('Loading Wine Quality dataset...')
    train_dataset = WineQualityDataset(split='train', train_ratio=0.8, random_state=42)
    test_dataset = WineQualityDataset(split='test', train_ratio=0.8, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.log(f'Train samples: {len(train_dataset)}')
    logger.log(f'Test samples: {len(test_dataset)}')
    logger.log('')

    # Initialize model
    input_size = 11  # Wine quality features
    output_size = 1  # Quality score
    model = SimpleMLP(input_size, hidden_size, output_size).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'Model: SimpleMLP')
    logger.log(f'Total parameters: {total_params:,}')
    logger.log('')

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # L1 regularizer
    l1_reg = L1Regularizer(lambda_l1=lambda_l1)

    # Log configuration
    logger.log_config({
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'hidden_size': hidden_size,
        'lambda_l1': lambda_l1,
        'optimizer': 'Adam',
        'loss': 'MSELoss'
    })

    # Training loop
    logger.log('=== Training ===')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_loss_base = 0.0
        train_loss_l1 = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            base_loss = criterion(outputs, y_batch)

            # Add L1 regularization
            l1_penalty = l1_reg(model)
            loss = base_loss + l1_penalty

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_base += base_loss.item()
            train_loss_l1 += l1_penalty.item()

        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_base_loss = train_loss_base / len(train_loader)
        avg_l1_loss = train_loss_l1 / len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        # Log metrics
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log_metrics(
                epoch=f'{epoch+1}/{epochs}',
                metrics_dict={
                    'Train Loss': avg_train_loss,
                    'Base': avg_base_loss,
                    'L1': avg_l1_loss,
                    'Test Loss': avg_test_loss
                }
            )

    logger.log('')
    logger.log('=== Training Complete ===')
    logger.log(f'Final Train Loss: {avg_train_loss:.4f}')
    logger.log(f'Final Test Loss: {avg_test_loss:.4f}')

    # Close logger
    logger.close()


if __name__ == '__main__':
    train_wine_mlp()
