# Implementation Plan - Phase 1

## Overview
This document outlines the implementation plan for the Learning Curves Regularization project using PyTorch and CIFAR-10 dataset.

## Phase 1.0: Basic Training Infrastructure (Current)

### Objectives
- Set up basic training pipeline with CIFAR-10
- Implement 3 model architectures: MLP, CNN, ResNet-18
- Train models and verify convergence
- Generate and save basic training plots
- **NO early stopping or double descent yet**

### Architecture Overview

```
curvas_aprendizado/
├── data/
│   └── cifar10_loader.py          # CIFAR-10 dataset loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── mlp.py                     # MLP architecture
│   ├── cnn.py                     # CNN architecture
│   └── resnet.py                  # ResNet-18 wrapper
├── training/
│   ├── __init__.py
│   └── trainer.py                 # Basic training loop
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                 # Evaluation metrics
├── visualization/
│   ├── __init__.py
│   └── plots.py                   # Plot generation
├── configs/
│   └── base_config.py             # Configuration management
├── results/                       # Generated output (timestamped folders)
├── main.py                        # Main entry point
├── requirements.txt
└── .venv/
```

## 1. Data Loading (data/cifar10_loader.py)

### CIFAR-10 Dataset Specifications
- **Training samples**: 50,000
- **Test samples**: 10,000
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image size**: 32x32 RGB

### Implementation

```python
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class CIFAR10DataModule:
    """
    CIFAR-10 data loading and preprocessing
    
    Reference: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
    """
    
    def __init__(self, data_dir='./data', batch_size=64, val_split=0.1, num_workers=2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        
        # CIFAR-10 normalization values
        # Mean and std computed on training set
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """
        Define preprocessing transforms
        Reference: https://docs.pytorch.org/vision/main/transforms.html
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def prepare_data(self):
        """Download CIFAR-10 dataset"""
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self):
        """Setup train, validation, and test datasets"""
        # Load full training set
        full_train = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=False
        )
        
        # Split into train and validation
        val_size = int(len(full_train) * self.val_split)
        train_size = len(full_train) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Load test set
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=False
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False  # Set to False for CPU
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
```

## 2. Model Architectures

### 2.1 MLP (models/mlp.py)

```python
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for CIFAR-10
    
    Input: 32x32x3 = 3072 features
    Output: 10 classes
    """
    
    def __init__(self, hidden_sizes=[512, 256], dropout=0.2):
        super(MLP, self).__init__()
        
        layers = []
        input_size = 32 * 32 * 3  # CIFAR-10 flattened
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, 10))
        
        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)
```

### 2.2 CNN (models/cnn.py)

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10
    Based on PyTorch CIFAR-10 tutorial
    
    Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    
    def __init__(self, num_filters=64):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, num_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
        # After 3 pooling layers: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(num_filters * 4 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.conv1(x)))
        
        # Conv block 2
        x = self.pool(self.relu(self.conv2(x)))
        
        # Conv block 3
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

### 2.3 ResNet-18 (models/resnet.py)

```python
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10
    
    Reference: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
    
    Note: Standard ResNet-18 expects 224x224 images. 
    For CIFAR-10 (32x32), we modify the first conv layer.
    """
    
    def __init__(self, num_classes=10, width_multiplier=1.0):
        super(ResNet18CIFAR, self).__init__()
        
        # Load ResNet-18 without pretrained weights
        self.model = resnet18(weights=None)
        
        # Modify first conv layer for CIFAR-10 (32x32 instead of 224x224)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # CIFAR-10: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Remove the max pooling layer (too aggressive for 32x32 images)
        self.model.maxpool = nn.Identity()
        
        # Modify final layer for CIFAR-10 (10 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Apply width multiplier (for double descent experiments later)
        if width_multiplier != 1.0:
            self._apply_width_multiplier(width_multiplier)
    
    def _apply_width_multiplier(self, multiplier):
        """Apply width multiplier to all layers (for future double descent)"""
        # This will be implemented for Phase 1.1
        pass
    
    def forward(self, x):
        return self.model(x)
```

## 3. Training Loop (training/trainer.py)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

class Trainer:
    """Basic training loop for PyTorch models"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        max_epochs=100
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        
        # Track metrics
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / len(pbar),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self):
        """Full training loop"""
        print(f"Training for {self.max_epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Track time
            epoch_time = time.time() - epoch_start
            
            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            
            # Print summary
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Time: {epoch_time:.2f}s")
        
        return self.history
    
    def test(self):
        """Test the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Testing'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        print(f"\nTest Accuracy: {test_acc:.2f}%")
        
        return test_acc
```

## 4. Visualization (visualization/plots.py)

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
from datetime import datetime

class ResultsVisualizer:
    """Generate and save training plots"""
    
    def __init__(self, output_dir='results'):
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Saving results to: {self.save_dir}")
    
    def plot_training_curves(self, history, model_name):
        """Plot loss and accuracy curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} - Loss Curves')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{model_name} - Accuracy Curves')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'{model_name}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")
    
    def plot_comparison(self, results_dict):
        """Compare multiple models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        for model_name, history in results_dict.items():
            epochs = range(1, len(history['val_loss']) + 1)
            ax1.plot(epochs, history['val_loss'], label=model_name)
            ax2.plot(epochs, history['val_acc'], label=model_name)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Model Comparison - Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Accuracy (%)')
        ax2.set_title('Model Comparison - Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")
```

## 5. Main Script (main.py)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from data.cifar10_loader import CIFAR10DataModule
from models.mlp import MLP
from models.cnn import SimpleCNN
from models.resnet import ResNet18CIFAR
from training.trainer import Trainer
from visualization.plots import ResultsVisualizer
import json
import os

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def main():
    # Set seed
    set_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("\n=== Loading CIFAR-10 ===")
    data_module = CIFAR10DataModule(batch_size=64, val_split=0.1)
    data_module.prepare_data()
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Visualizer
    visualizer = ResultsVisualizer()
    
    # Models to train
    models_config = {
        'MLP': MLP(hidden_sizes=[512, 256]),
        'CNN': SimpleCNN(num_filters=64),
        'ResNet18': ResNet18CIFAR()
    }
    
    results = {}
    test_accuracies = {}
    
    # Train each model
    for model_name, model in models_config.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_epochs=50  # Start with 50 epochs
        )
        
        # Train
        history = trainer.train()
        
        # Test
        test_acc = trainer.test()
        
        # Save results
        results[model_name] = history
        test_accuracies[model_name] = test_acc
        
        # Plot individual results
        visualizer.plot_training_curves(history, model_name)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(visualizer.save_dir, f'{model_name}_best.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model: {checkpoint_path}")
    
    # Comparison plots
    print("\n=== Generating Comparison Plots ===")
    visualizer.plot_comparison(results)
    
    # Save summary
    summary = {
        'test_accuracies': test_accuracies,
        'training_config': {
            'batch_size': 64,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'epochs': 50,
            'val_split': 0.1
        }
    }
    
    summary_path = os.path.join(visualizer.save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary: {summary_path}")
    print("\n=== Training Complete ===")

if __name__ == '__main__':
    main()
```

## 6. Requirements (requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

## Implementation Steps

1. ✅ Create project structure
2. ✅ Implement data loading (CIFAR-10)
3. ✅ Implement models (MLP, CNN, ResNet-18)
4. ✅ Implement training loop
5. ✅ Implement visualization
6. ✅ Create main script
7. ⏳ Run training and verify convergence
8. ⏳ Analyze results

## Expected Outputs

After running `main.py`, the `results/YYYYMMDD_HHMMSS/` folder will contain:
- `MLP_training_curves.png`
- `CNN_training_curves.png`
- `ResNet18_training_curves.png`
- `model_comparison.png`
- `MLP_best.pth`
- `CNN_best.pth`
- `ResNet18_best.pth`
- `summary.json`

## Next Phases

- **Phase 1.1**: Early Stopping Implementation
- **Phase 1.2**: Double Descent Experiments
- **Phase 2**: Jupyter Notebooks

## Notes

- All models use the same hyperparameters initially for fair comparison
- CPU-optimized (no CUDA optimizations)
- Timestamped outputs for tracking experiments
- Reproducible with fixed random seeds

