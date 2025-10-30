"""
Dataset utilities for visualization and exploration

Provides functions for loading, visualizing, and analyzing datasets
used in the training pipeline.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import random


def get_class_names(dataset_name):
    """
    Get class names for a dataset
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        list: List of class names
    """
    class_mappings = {
        'FashionMNIST': [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ],
        'CIFAR10': [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        'SVHN': [str(i) for i in range(10)]  # Digits 0-9
    }
    
    return class_mappings.get(dataset_name, [f'Class {i}' for i in range(10)])


def get_random_samples(dataset, num_samples=16, seed=42):
    """
    Get random samples from a dataset
    
    Args:
        dataset: PyTorch dataset
        num_samples (int): Number of samples to return
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (images, labels) tensors
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Get random indices
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    # Collect samples
    images = []
    labels = []
    
    for idx in indices:
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)
    
    return torch.stack(images), torch.tensor(labels)


def plot_samples(images, labels, class_names, grid_size=(4, 4), figsize=(10, 10)):
    """
    Plot a grid of sample images with their labels
    
    Args:
        images (torch.Tensor): Tensor of images (N, C, H, W)
        labels (torch.Tensor): Tensor of labels
        class_names (list): List of class names
        grid_size (tuple): Grid dimensions (rows, cols)
        figsize (tuple): Figure size
    """
    num_samples = len(images)
    rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(min(num_samples, rows * cols)):
        ax = axes[i]
        
        # Convert image tensor to numpy for plotting
        if images[i].dim() == 3:  # (C, H, W)
            img = images[i].permute(1, 2, 0)  # (H, W, C)
            if img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)  # (H, W)
        else:  # (H, W)
            img = images[i]
        
        # Denormalize if needed (assuming ImageNet normalization)
        if img.min() < 0:  # Likely normalized
            img = (img + 1) / 2  # Rough denormalization
            img = torch.clamp(img, 0, 1)
        
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(f'{class_names[labels[i]]}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def get_dataset_stats(data_module):
    """
    Get statistics about a dataset
    
    Args:
        data_module: Dataset module with train/val/test datasets
        
    Returns:
        dict: Dictionary with dataset statistics
    """
    stats = {
        'dataset_name': data_module.name,
        'train_size': len(data_module.train_dataset),
        'val_size': len(data_module.val_dataset),
        'test_size': len(data_module.test_dataset),
        'total_size': len(data_module.train_dataset) + len(data_module.val_dataset) + len(data_module.test_dataset),
        'num_classes': len(get_class_names(data_module.name)),
        'class_names': get_class_names(data_module.name)
    }
    
    # Add data cap info if available
    if hasattr(data_module, 'data_cap_rate'):
        stats['data_cap_rate'] = data_module.data_cap_rate
        stats['original_train_size'] = stats['train_size'] * data_module.data_cap_rate
    
    return stats


def plot_class_distribution(data_module, figsize=(12, 4)):
    """
    Plot class distribution for train/val/test sets
    
    Args:
        data_module: Dataset module with train/val/test datasets
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    datasets = [
        (data_module.train_dataset, 'Training'),
        (data_module.val_dataset, 'Validation'),
        (data_module.test_dataset, 'Test')
    ]
    
    class_names = get_class_names(data_module.name)
    
    for i, (dataset, title) in enumerate(datasets):
        # Count class occurrences
        class_counts = [0] * len(class_names)
        for _, label in dataset:
            class_counts[label] += 1
        
        # Plot
        axes[i].bar(range(len(class_names)), class_counts)
        axes[i].set_title(f'{title} Set\n({len(dataset)} samples)')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Count')
        axes[i].set_xticks(range(len(class_names)))
        axes[i].set_xticklabels(class_names, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_dataset_summary(data_module):
    """
    Print a summary of dataset statistics
    
    Args:
        data_module: Dataset module with train/val/test datasets
    """
    stats = get_dataset_stats(data_module)
    
    print("=" * 60)
    print(f"  Dataset: {stats['dataset_name']}")
    print("=" * 60)
    print(f"Classes: {stats['num_classes']}")
    print(f"Class names: {', '.join(stats['class_names'])}")
    print()
    print("Dataset sizes:")
    print(f"  Training:   {stats['train_size']:,} samples")
    print(f"  Validation: {stats['val_size']:,} samples")
    print(f"  Test:       {stats['test_size']:,} samples")
    print(f"  Total:      {stats['total_size']:,} samples")
    
    if 'data_cap_rate' in stats:
        print(f"\nData cap rate: {stats['data_cap_rate']} (using 1/{stats['data_cap_rate']} of data)")
        print(f"Original training size would be: {stats['original_train_size']:,} samples")
    
    print("=" * 60)


def visualize_dataset_samples(data_module, num_samples=16, seed=42):
    """
    Complete dataset visualization: stats + samples + class distribution
    
    Args:
        data_module: Dataset module with train/val/test datasets
        num_samples (int): Number of samples to display
        seed (int): Random seed for reproducibility
    """
    # Print summary
    print_dataset_summary(data_module)
    
    # Get and plot random samples
    print(f"\nRandom samples from training set:")
    images, labels = get_random_samples(data_module.train_dataset, num_samples, seed)
    class_names = get_class_names(data_module.name)
    plot_samples(images, labels, class_names)
    
    # Plot class distribution
    print(f"\nClass distribution:")
    plot_class_distribution(data_module)
