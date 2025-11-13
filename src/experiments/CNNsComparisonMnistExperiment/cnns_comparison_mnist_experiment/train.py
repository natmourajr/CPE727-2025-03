import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd

# Add paths for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../modules'))

import torch
from torch import nn, optim
from tqdm import tqdm

# Import mnist loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../dataloaders/MnistLoader'))
from mnist_loader.loader import build_loaders as MnistLoader

from EfficientNet.efficientnet import EfficientNetB0
from AlexNet.alexnet import AlexNet

# Import logger
from logger import ExperimentLogger

MODELS_NAME = {
    'efficientnet_b0': EfficientNetB0,
    'alexnet': AlexNet,
}

def train_and_compare_cnns_mnist(
    models = ['alexnet', 'efficientnet_b0'],
    epochs=30,
    batch_size=16,
    learning_rate=0.001,
    dataset_fraction=None,
    device=None,
    transfer_learning=False
):
    
    # Initialize logger
    repo_root = os.path.join(os.path.dirname(__file__), '../../../..')
    results_dir = os.path.abspath(os.path.join(repo_root, 'results'))
    logger = ExperimentLogger('cnn_mnist', results_dir=results_dir)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f'Using device: {device}')

    for model_name in models:
        if model_name not in MODELS_NAME:
            logger.log(f'Unknown model: {model_name}. Skipping...')
            continue

        # Set Model
        model = MODELS_NAME[model_name](num_channels=1, num_classes=10, device=device, pretrained=transfer_learning)

        # Set Dataloaders
        train_loader, test_loader = MnistLoader(
            root=os.path.join(repo_root, 'tmp', 'datasets'),
            transforms=model.default_dataloader_transforms(),
            batch_size=batch_size,
            dataset_fraction=dataset_fraction
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Log configuration
        logger.log_config({
            'model': model_name,
            'transfer_learning': transfer_learning,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'dataset_fraction': dataset_fraction,
            'optimizer': 'Adam',
            'loss': 'CrossEntropyLoss'
        })

        path = os.path.join(repo_root, 'tmp', 'data', 'mnist', model_name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'training.csv'), 'w') as f:
            f.write('epoch,train_loss,test_loss,accuracy,timedelta\n')

        # Training loop
        logger.log('=== Training ===')
        for epoch in tqdm(range(epochs)):
            model.train()
            train_loss = 0.0
            start_time = time.time()

            for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_loader)):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

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
                for batch_idx, (X_batch, y_batch) in tqdm(enumerate(test_loader)):
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    test_loss += loss.item()

                    # Calculate accuracy
                    _, predictions = torch.max(outputs, 1)
                    correct += (predictions == y_batch).sum().item()
                    total += y_batch.size(0)

            avg_test_loss = test_loss / len(test_loader)
            accuracy = 100 * correct / total
            epoch_time = time.time() - start_time

            # Log metrics
            logger.log_metrics(
                epoch=f'{epoch+1}/{epochs}',
                metrics_dict={
                    'Train Loss': avg_train_loss,
                    'Test Loss': avg_test_loss,
                    'Accuracy (%)': accuracy,
                    'Epoch Time (s)': epoch_time
                }
            )
            with open(os.path.join(path, 'training.csv'), 'a+') as f:
                f.write(f'{epoch+1},{avg_train_loss},{avg_test_loss},{accuracy},{epoch_time}\n')

        epoch_time = time.time() - start_time
        logger.log('')
        logger.log('=== Training Complete ===')
        logger.log(f'Final Train Loss: {avg_train_loss:.4f}')
        logger.log(f'Final Test Loss: {avg_test_loss:.4f}')
        logger.log(f'Final Test Accuracy: {accuracy:.2f}%')
        with open(os.path.join(path, 'training.csv'), 'a+') as f:
            f.write(f'final,{avg_train_loss},{avg_test_loss},{accuracy},{epoch_time}\n')

    generate_grafs(models)
    # Close logger()
    logger.close()

def generate_grafs(model_names = ['alexnet', 'efficientnet_b0']):
    repo_root = os.path.join(os.path.dirname(__file__), '../../../..')
    model_dfs = {}
    for model_name in model_names:
        path = os.path.join(repo_root, 'tmp', 'data', 'mnist', model_name, 'training.csv')
        df = pd.read_csv(path)
        model_dfs[model_name] = df

    columns = set()
    for df in model_dfs.values():
        columns.update(df.columns)

    for col in columns:
        if col == "epoch":
            continue

        plt.figure(figsize=(8, 5))
        for model_name, df in model_dfs.items():
            if "epoch" not in df.columns or col not in df.columns:
                continue
            plt.plot(df["epoch"], df[col], label=model_name, linewidth=2)

        plt.title(col.replace("_", " ").title(), fontsize=14, weight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()

        # Caminho de salvamento
        filename = f"{col}_{'_'.join([name.replace('_', '') for name in model_names])}.png"
        filepath = os.path.join(repo_root, 'tmp', 'data', 'mnist', filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == '__main__':
    train_and_compare_cnns_mnist()