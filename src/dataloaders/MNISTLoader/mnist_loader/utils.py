"""Utility functions for MNIST data loaders."""
import torch
from torch.utils.data import Subset
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


def generate_cv_splits(dataset, n_splits=5, shuffle=True, random_state=42, stratified=True):
    """
    Generate cross-validation splits for MNIST or Noisy MNIST datasets.

    Args:
        dataset: MNISTDataset or NoisyMNISTDataset instance
        n_splits (int): Number of folds for cross-validation (default: 5)
        shuffle (bool): Whether to shuffle the data before splitting (default: True)
        random_state (int): Random seed for reproducibility (default: 42)
        stratified (bool): Whether to use stratified splits to preserve class distribution (default: True)

    Returns:
        list of tuples: Each tuple contains (train_dataset, val_dataset) for one fold

    Example:
        >>> from mnist_loader.loader import MNISTDataset
        >>> dataset = MNISTDataset(download_path="./data", train=True)
        >>> cv_splits = generate_cv_splits(dataset, n_splits=5)
        >>> for fold, (train_data, val_data) in enumerate(cv_splits):
        >>>     print(f"Fold {fold}: Train size={len(train_data)}, Val size={len(val_data)}")
    """
    # Get labels for stratification
    if hasattr(dataset, 'y_classes'):
        # NoisyMNISTDataset has y_classes attribute
        labels = dataset.y_classes.numpy()
    elif hasattr(dataset, 'y'):
        # MNISTDataset has y attribute
        labels = dataset.y.numpy()
    else:
        raise ValueError("Dataset must have 'y' or 'y_classes' attribute")

    # Choose splitter
    if stratified:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Generate splits
    cv_splits = []
    indices = np.arange(len(dataset))

    for _, (train_idx, test_idx) in enumerate(kfold.split(indices, labels)):
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, test_idx.tolist())
        cv_splits.append((train_subset, val_subset))

    return cv_splits


def export_cv_splits_to_csv(dataset, cv_splits, output_path,):
    """
    Export cross-validation splits to a CSV file with a split column.

    Args:
        dataset: Original dataset (MNISTDataset or NoisyMNISTDataset)
        cv_splits: List of (train_subset, val_subset) tuples from generate_cv_splits
        output_path (str): Path to save the CSV file
        include_data (bool): If True, includes flattened image data (default: False)

    Returns:
        pandas.DataFrame: The generated DataFrame

    Example:
        >>> dataset = MNISTDataset(download_path="./data", train=True)
        >>> cv_splits = generate_cv_splits(dataset, n_splits=5)
        >>> df = export_cv_splits_to_csv(dataset, cv_splits, "cv_splits.csv")
    """
    import pandas as pd
    from pathlib import Path

    # Get labels and data
    if hasattr(dataset, 'y_classes'):
        labels = dataset.y_classes.numpy()
    elif hasattr(dataset, 'y'):
        labels = dataset.y.numpy()
    else:
        raise ValueError("Dataset must have 'y' or 'y_classes' attribute")

    # Initialize data structure
    data_dict = {
        'index': [],
        'label': [],
        'split': [],
    }

    # Add fold information
    for fold_idx, (train_subset, _) in enumerate(cv_splits):
        # Add training indices
        for idx in train_subset.indices:
            data_dict['index'].append(idx)
            data_dict['label'].append(int(labels[idx]))
            data_dict['split'].append(fold_idx)

    # Create DataFrame
    df = pd.DataFrame(data_dict)

    # Sort by index for consistency
    df = df.sort_values(['index', 'split']).reset_index(drop=True)

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df