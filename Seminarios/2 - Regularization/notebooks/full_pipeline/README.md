# Full Regularization Pipeline - Jupyter Notebook Instructions

## Overview

This notebook implements a comprehensive regularization study using Fashion MNIST dataset with Convolutional Neural Networks (CNNs). The pipeline systematically compares different regularization techniques both individually and in combination to understand their effectiveness in preventing overfitting.

**Key Features:**
- Implements all major regularization methods (L1, L2, Elastic Net, data augmentation, Dropout, Batch Normalization, Early Stopping)
- Provides comprehensive visualization and analysis tools
- Modular design with separate Python modules for clean code organization

## Objectives

1. **Demonstrate Overfitting**: Train a CNN on a reduced Fashion MNIST dataset to show clear overfitting
2. **Individual Regularization Comparison**: Test each regularization technique separately
3. **Combined Regularization Analysis**: Stack multiple regularization methods to find optimal combinations
4. **Visualization**: Create comprehensive plots showing training/validation curves and performance metrics

## Dataset and Model Setup

### Fashion MNIST Configuration
- **Dataset**: Fashion MNIST (28x28 grayscale images, 10 classes)
- **Data Reduction**: Use `data_cap_rate=20` (1/20 of original data) to force overfitting
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

### CNN Architecture
```python
class FashionMNISTCNN(nn.Module):
    def __init__(self, num_filters=64, dropout=0.25, use_batch_norm=True):
        super().__init__()
        # Conv layers: 3 blocks with increasing filters
        self.conv1 = nn.Conv2d(1, num_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, 3, padding=1)
        self.conv3 = nn.Conv2d(num_filters*2, num_filters*4, 3, padding=1)
        
        # Batch normalization (optional)
        self.bn1 = nn.BatchNorm2d(num_filters) if use_batch_norm else None
        self.bn2 = nn.BatchNorm2d(num_filters*2) if use_batch_norm else None
        self.bn3 = nn.BatchNorm2d(num_filters*4) if use_batch_norm else None
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # FC layers
        fc_input_size = num_filters * 4 * 3 * 3  # After 3 pooling layers: 28->14->7->3
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, 10)
```

## Experiment Structure

### Phase 1: Baseline Overfitting Demonstration
**Objective**: Establish baseline with clear overfitting

```python
# Configuration for overfitting
config_baseline = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'max_epochs': 100,
    'data_cap_rate': 20,  # Use only 1/20 of data
    'early_stopping_enabled': False,  # Train full epochs
    'regularization': None
}

# Train baseline CNN (no regularization)
baseline_model = FashionMNISTCNN(dropout=0.0, use_batch_norm=False)
baseline_history = train_model(baseline_model, config_baseline)
```

**Expected Results**: 
- Training accuracy: ~95-98%
- Validation accuracy: ~60-70%
- Clear gap indicating overfitting

### Phase 2: Individual Regularization Techniques

#### 2.1 L1 Regularization
```python
class L1Regularizer:
    def __init__(self, lambda_l1=0.01):
        self.lambda_l1 = lambda_l1
    
    def __call__(self, model):
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        return self.lambda_l1 * l1_loss

# Training with L1
config_l1 = config_baseline.copy()
config_l1['regularization'] = L1Regularizer(lambda_l1=0.01)
l1_model = FashionMNISTCNN(dropout=0.0, use_batch_norm=False)
l1_history = train_model(l1_model, config_l1)
```

#### 2.2 L2 Regularization (Weight Decay)
```python
class L2Regularizer:
    def __init__(self, lambda_l2=0.01):
        self.lambda_l2 = lambda_l2
    
    def __call__(self, model):
        l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())
        return self.lambda_l2 * l2_loss

# Training with L2
config_l2 = config_baseline.copy()
config_l2['regularization'] = L2Regularizer(lambda_l2=0.01)
l2_model = FashionMNISTCNN(dropout=0.0, use_batch_norm=False)
l2_history = train_model(l2_model, config_l2)
```

#### 2.3 Elastic Net Regularization
```python
class ElasticNetRegularizer:
    def __init__(self, lambda_l1=0.005, lambda_l2=0.005):
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
    
    def __call__(self, model):
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())
        return self.lambda_l1 * l1_loss + self.lambda_l2 * l2_loss
```

#### 2.4 Dropout Regularization
```python
# Training with Dropout
config_dropout = config_baseline.copy()
config_dropout['regularization'] = None  # Dropout is in model architecture
dropout_model = FashionMNISTCNN(dropout=0.5, use_batch_norm=False)
dropout_history = train_model(dropout_model, config_dropout)
```

#### 2.5 Batch Normalization
```python
# Training with Batch Normalization
config_bn = config_baseline.copy()
config_bn['regularization'] = None  # BN is in model architecture
bn_model = FashionMNISTCNN(dropout=0.0, use_batch_norm=True)
bn_history = train_model(bn_model, config_bn)
```

#### 2.6 Early Stopping
```python
# Training with Early Stopping
config_early_stop = config_baseline.copy()
config_early_stop['early_stopping_enabled'] = True
config_early_stop['early_stopping_patience'] = 10
early_stop_model = FashionMNISTCNN(dropout=0.0, use_batch_norm=False)
early_stop_history = train_model(early_stop_model, config_early_stop)
```

#### 2.7 Data Augmentation (Albumentations)

**Why Albumentations?**
- **More transforms**: 70+ augmentation techniques vs ~20 in torchvision
- **Better performance**: Optimized C++ backend with OpenCV
- **Flexible probability control**: Each transform has individual probability settings
- **Advanced techniques**: Elastic deformation, perspective transforms, advanced noise types
- **Better API**: More intuitive and consistent interface

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Data augmentation transforms using Albumentations
augmentation_transforms = A.Compose([
    A.Rotate(limit=15, p=0.7),  # Random rotation up to 15 degrees
    A.HorizontalFlip(p=0.5),    # Horizontal flip
    A.ShiftScaleRotate(
        shift_limit=0.1,        # Shift up to 10% of image size
        scale_limit=0.1,        # Scale between 0.9-1.1
        rotate_limit=10,        # Rotate up to 10 degrees
        p=0.6
    ),
    A.ElasticTransform(
        alpha=1, sigma=50, alpha_affine=50, p=0.3
    ),  # Elastic deformation
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add Gaussian noise
    A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.5
    ),  # Brightness/contrast adjustment
    A.Normalize(mean=(0.2860,), std=(0.3530,)),  # FashionMNIST normalization
    ToTensorV2()  # Convert to tensor
])

# Custom dataset class for Albumentations
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform:
            # Convert tensor back to numpy for albumentations
            if isinstance(image, torch.Tensor):
                image = image.numpy().transpose(1, 2, 0)  # CHW to HWC
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

# Training with Data Augmentation
config_aug = config_baseline.copy()
config_aug['augmentation_transform'] = augmentation_transforms
aug_model = FashionMNISTCNN(dropout=0.0, use_batch_norm=False)
aug_history = train_model(aug_model, config_aug)
```

### Phase 3: Combined Regularization Techniques

#### 3.1 Best Individual Combinations
```python
# Combine top-performing individual techniques
config_combined_1 = config_baseline.copy()
config_combined_1['regularization'] = L2Regularizer(lambda_l2=0.01)
config_combined_1['early_stopping_enabled'] = True
combined_1_model = FashionMNISTCNN(dropout=0.3, use_batch_norm=True)
combined_1_history = train_model(combined_1_model, config_combined_1)
```

#### 3.2 All Regularization Stack
```python
# Maximum regularization stack
config_max_reg = config_baseline.copy()
config_max_reg['regularization'] = ElasticNetRegularizer(lambda_l1=0.005, lambda_l2=0.005)
config_max_reg['early_stopping_enabled'] = True
config_max_reg['augmentation_transform'] = augmentation_transforms
max_reg_model = FashionMNISTCNN(dropout=0.4, use_batch_norm=True)
max_reg_history = train_model(max_reg_model, config_max_reg)
```

## Training Pipeline Implementation

### Core Training Function
```python
def train_model(model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train a single model with given configuration
    
    Args:
        model: CNN model to train
        config: Training configuration dictionary
        device: Device to train on
    
    Returns:
        history: Dictionary with training metrics
    """
    # Setup data
    data_module = FashionMNISTDataModule(
        batch_size=config['batch_size'],
        data_cap_rate=config['data_cap_rate']
    )
    data_module.prepare_data()
    data_module.setup()
    
    # Apply augmentation if specified
    if 'augmentation_transform' in config:
        train_dataset = AlbumentationsDataset(
            data_module.train_dataset, 
            config['augmentation_transform']
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2
        )
    else:
        train_loader = data_module.train_dataloader()
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Setup early stopping if enabled
    early_stopping = None
    if config.get('early_stopping_enabled', False):
        early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            mode='min',
            restore_best_weights=True
        )
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(config['max_epochs']):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, 
                                           criterion, optimizer, device, config)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, data_module.val_dataloader(), 
                                          criterion, device)
        
        # Record metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if early_stopping:
            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return history
```

### Loss Function with Regularization
```python
def compute_loss_with_regularization(model, outputs, targets, criterion, regularization=None):
    """
    Compute loss including regularization terms
    
    Args:
        model: The neural network model
        outputs: Model predictions
        targets: Ground truth labels
        criterion: Base loss function (e.g., CrossEntropyLoss)
        regularization: Regularization object (L1, L2, ElasticNet)
    
    Returns:
        total_loss: Combined loss + regularization
    """
    base_loss = criterion(outputs, targets)
    
    if regularization is not None:
        reg_loss = regularization(model)
        total_loss = base_loss + reg_loss
    else:
        total_loss = base_loss
    
    return total_loss
```

## Visualization and Analysis

### 1. Training Curves Comparison
```python
def plot_training_curves(histories, labels, save_path=None):
    """
    Plot training and validation curves for multiple models
    
    Args:
        histories: List of history dictionaries
        labels: List of model labels
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Loss
    axes[0,0].set_title('Training Loss')
    for history, label in zip(histories, labels):
        axes[0,0].plot(history['train_loss'], label=label)
    axes[0,0].legend()
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    
    # Validation Loss
    axes[0,1].set_title('Validation Loss')
    for history, label in zip(histories, labels):
        axes[0,1].plot(history['val_loss'], label=label)
    axes[0,1].legend()
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    
    # Training Accuracy
    axes[1,0].set_title('Training Accuracy')
    for history, label in zip(histories, labels):
        axes[1,0].plot(history['train_acc'], label=label)
    axes[1,0].legend()
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy')
    
    # Validation Accuracy
    axes[1,1].set_title('Validation Accuracy')
    for history, label in zip(histories, labels):
        axes[1,1].plot(history['val_acc'], label=label)
    axes[1,1].legend()
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

### 2. Performance Comparison Table
```python
def create_performance_table(histories, labels):
    """
    Create a comparison table of final performance metrics
    
    Args:
        histories: List of history dictionaries
        labels: List of model labels
    
    Returns:
        pandas.DataFrame: Performance comparison table
    """
    results = []
    
    for history, label in zip(histories, labels):
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        # Calculate overfitting gap
        overfitting_gap = final_train_acc - final_val_acc
        
        results.append({
            'Model': label,
            'Final Train Acc': f"{final_train_acc:.3f}",
            'Final Val Acc': f"{final_val_acc:.3f}",
            'Overfitting Gap': f"{overfitting_gap:.3f}",
            'Final Train Loss': f"{final_train_loss:.3f}",
            'Final Val Loss': f"{final_val_loss:.3f}"
        })
    
    return pd.DataFrame(results)
```

### 3. Regularization Effectiveness Analysis
```python
def analyze_regularization_effectiveness(results_df):
    """
    Analyze which regularization techniques are most effective
    
    Args:
        results_df: Performance comparison DataFrame
    
    Returns:
        dict: Analysis results
    """
    analysis = {}
    
    # Find best validation accuracy
    best_val_acc_idx = results_df['Final Val Acc'].astype(float).idxmax()
    analysis['best_model'] = results_df.loc[best_val_acc_idx, 'Model']
    analysis['best_val_acc'] = results_df.loc[best_val_acc_idx, 'Final Val Acc']
    
    # Find least overfitting
    least_overfitting_idx = results_df['Overfitting Gap'].astype(float).idxmin()
    analysis['least_overfitting_model'] = results_df.loc[least_overfitting_idx, 'Model']
    analysis['least_overfitting_gap'] = results_df.loc[least_overfitting_idx, 'Overfitting Gap']
    
    # Rank by validation accuracy
    analysis['val_acc_ranking'] = results_df.sort_values('Final Val Acc', ascending=False)['Model'].tolist()
    
    return analysis
```

## Notebook Structure

### Cell 1: Imports and Setup
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Cell 2: Data Module Implementation
```python
# Import or implement FashionMNISTDataModule
# (Use existing implementation from curvas_aprendizado/data/)
```

### Cell 3: Model Architecture
```python
# Implement FashionMNISTCNN class
# (Adapt from existing SimpleCNN implementation)
```

### Cell 4: Regularization Classes
```python
# Implement all regularization classes
# (Use existing implementations from src/modules/regularization/)
```

### Cell 5: Training Pipeline
```python
# Implement training functions
# train_model(), train_epoch(), validate_epoch()
```

### Cell 6: Phase 1 - Baseline Overfitting
```python
# Train baseline model and visualize overfitting
baseline_history = train_model(baseline_model, config_baseline)
plot_training_curves([baseline_history], ['Baseline (No Regularization)'])
```

### Cell 7: Phase 2 - Individual Regularization
```python
# Train all individual regularization models
individual_histories = []
individual_labels = []

# L1 Regularization
l1_history = train_model(l1_model, config_l1)
individual_histories.append(l1_history)
individual_labels.append('L1 Regularization')

# L2 Regularization
l2_history = train_model(l2_model, config_l2)
individual_histories.append(l2_history)
individual_labels.append('L2 Regularization')

# ... (continue for all techniques)

# Plot comparison
plot_training_curves(individual_histories, individual_labels)
```

### Cell 8: Individual Results Analysis
```python
# Create performance table and analysis
individual_results = create_performance_table(individual_histories, individual_labels)
print(individual_results)

individual_analysis = analyze_regularization_effectiveness(individual_results)
print(f"Best individual technique: {individual_analysis['best_model']}")
print(f"Least overfitting: {individual_analysis['least_overfitting_model']}")
```

### Cell 9: Phase 3 - Combined Regularization
```python
# Train combined regularization models
combined_histories = []
combined_labels = []

# Best individual combinations
combined_1_history = train_model(combined_1_model, config_combined_1)
combined_histories.append(combined_1_history)
combined_labels.append('L2 + Dropout + BatchNorm + EarlyStop')

# Maximum regularization
max_reg_history = train_model(max_reg_model, config_max_reg)
combined_histories.append(max_reg_history)
combined_labels.append('All Regularization Techniques')

# Plot comparison
plot_training_curves(combined_histories, combined_labels)
```

### Cell 10: Final Analysis and Conclusions
```python
# Comprehensive analysis
all_histories = [baseline_history] + individual_histories + combined_histories
all_labels = ['Baseline'] + individual_labels + combined_labels

final_results = create_performance_table(all_histories, all_labels)
print("Complete Performance Comparison:")
print(final_results)

# Final analysis
final_analysis = analyze_regularization_effectiveness(final_results)
print(f"\nBest overall model: {final_analysis['best_model']}")
print(f"Best validation accuracy: {final_analysis['best_val_acc']}")
print(f"Least overfitting model: {final_analysis['least_overfitting_model']}")
print(f"Smallest overfitting gap: {final_analysis['least_overfitting_gap']}")

# Create final visualization
plot_training_curves(all_histories, all_labels)
```

## Expected Outcomes

### Key Findings to Demonstrate:
1. **Clear Overfitting**: Baseline model shows significant train/validation gap
2. **Regularization Effectiveness**: Each technique reduces overfitting to different degrees
3. **Technique Ranking**: L2, Dropout, and Early Stopping typically most effective
4. **Combination Benefits**: Stacking techniques often provides best results
5. **Trade-offs**: Regularization may reduce training accuracy but improve generalization

### Performance Expectations:
- **Baseline**: Train Acc ~95%, Val Acc ~65% (30% gap)
- **Best Individual**: Train Acc ~85%, Val Acc ~80% (5% gap)
- **Best Combined**: Train Acc ~80%, Val Acc ~82% (2% gap)

## File Organization

```
full_pipeline/
├── README.md (this file)
├── regularization_pipeline.ipynb (main notebook)
├── models/
│   ├── __init__.py
│   └── cnn.py (FashionMNISTCNN implementation)
├── regularization/
│   ├── __init__.py
│   ├── l1.py (L1Regularizer)
│   ├── l2.py (L2Regularizer)
│   ├── elastic_net.py (ElasticNetRegularizer)
│   └── early_stopping.py (EarlyStopping)
├── training/
│   ├── __init__.py
│   ├── trainer.py (training pipeline)
│   └── utils.py (helper functions)
├── visualization/
│   ├── __init__.py
│   └── plots.py (plotting functions)
└── results/
    ├── models/ (saved model weights)
    ├── plots/ (generated plots)
    └── tables/ (performance tables)
```

## Dependencies

```python
# requirements.txt
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
pandas>=1.3.0
numpy>=1.21.0
seaborn>=0.11.0
scikit-learn>=1.0.0
albumentations>=1.3.0
opencv-python>=4.5.0
jupyter>=1.0.0
```

This comprehensive pipeline will provide a thorough understanding of regularization techniques and their effectiveness in preventing overfitting in deep learning models.