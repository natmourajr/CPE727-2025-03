"""Adapters for converting PyTorch datasets to pandas DataFrames

Dataset-agnostic conversion that works with any PyTorch Dataset.
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List


def dataset_to_dataframe(
    dataset: Dataset,
    feature_names: Optional[List[str]] = None,
    target_name: str = "target",
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """Convert any PyTorch Dataset to pandas DataFrame

    Works with any dataset that implements __len__ and __getitem__.

    Args:
        dataset: PyTorch Dataset instance
        feature_names: List of feature column names (auto-generated if None)
        target_name: Name for target column (default: 'target')
        max_samples: Maximum number of samples to convert (None = all)

    Returns:
        pandas DataFrame with features and target
    """
    n_samples = len(dataset)
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)

    data = []

    for i in range(n_samples):
        sample = dataset[i]

        # Handle different return formats
        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            X, y = sample
        else:
            X = sample
            y = None

        # Convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        # Flatten if needed
        if X.ndim > 1:
            X = X.flatten()

        # Build row
        row = list(X)

        if y is not None:
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            # Handle scalar or array targets
            if hasattr(y, 'size'):
                row.append(y.item() if y.size == 1 else y[0] if len(y.shape) == 1 else y)
            else:
                row.append(y)

        data.append(row)

    # Generate column names
    if len(data) > 0:
        n_features = len(data[0]) - (1 if y is not None else 0)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(
                f"Number of feature_names ({len(feature_names)}) doesn't match "
                f"number of features ({n_features})"
            )

        columns = feature_names + ([target_name] if y is not None else [])
    else:
        columns = []

    return pd.DataFrame(data, columns=columns)


def get_dataset_info(dataset: Dataset) -> dict:
    """Extract basic information from a dataset

    Args:
        dataset: PyTorch Dataset instance

    Returns:
        Dictionary with dataset metadata
    """
    sample = dataset[0]

    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        X, y = sample
    else:
        X = sample
        y = None

    info = {
        "n_samples": len(dataset),
        "has_targets": y is not None,
    }

    if isinstance(X, torch.Tensor):
        info["feature_shape"] = tuple(X.shape)
        info["feature_dtype"] = str(X.dtype)
        info["n_features"] = X.numel() if X.ndim == 1 else np.prod(X.shape)

    if y is not None and isinstance(y, torch.Tensor):
        info["target_shape"] = tuple(y.shape)
        info["target_dtype"] = str(y.dtype)

    return info
