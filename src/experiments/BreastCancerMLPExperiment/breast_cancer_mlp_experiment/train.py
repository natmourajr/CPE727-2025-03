import sys
import os

# Add paths for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../modules'))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

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


def train_breast_cancer_mlp(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    hidden_size=64,
    feature_strategy='onehot',
    target_strategy='binary',
    handle_missing='drop',
    device='cpu'
):
    """Train SimpleMLP on Breast Cancer dataset with preprocessing

    Args:
        epochs: Number of training epochs (default: 50)
        batch_size: Batch size for training (default: 32)
        learning_rate: Learning rate for optimizer (default: 0.001)
        hidden_size: Size of hidden layer (default: 64)
        feature_strategy: Feature encoding strategy ('onehot', 'label', 'ordinal')
        target_strategy: Target encoding strategy ('binary', 'label')
        handle_missing: Missing value strategy ('drop', 'most_frequent', 'constant')
        device: Device to train on ('cpu' or 'cuda')
    """
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

    # Log configuration
    logger.log_config({
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'hidden_size': hidden_size,
        'feature_strategy': feature_strategy,
        'target_strategy': target_strategy,
        'handle_missing': handle_missing,
        'optimizer': 'Adam',
        'loss': 'BCEWithLogitsLoss'
    })

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

            train_loss += loss.item()

        # Average loss
        avg_train_loss = train_loss / len(train_loader)

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

        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total

        # Log metrics
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log_metrics(
                epoch=f'{epoch+1}/{epochs}',
                metrics_dict={
                    'Train Loss': avg_train_loss,
                    'Test Loss': avg_test_loss,
                    'Accuracy (%)': accuracy
                }
            )

    logger.log('')
    logger.log('=== Training Complete ===')
    logger.log(f'Final Train Loss: {avg_train_loss:.4f}')
    logger.log(f'Final Test Loss: {avg_test_loss:.4f}')
    logger.log(f'Final Test Accuracy: {accuracy:.2f}%')

    # Close logger
    logger.close()


if __name__ == '__main__':
    train_breast_cancer_mlp()
