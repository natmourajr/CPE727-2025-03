"""Auto-detection of dataset types

Automatically determines if a dataset contains tabular, image, text, or other data.
"""
import torch
from torch.utils.data import Dataset


def detect_data_type(dataset: Dataset) -> str:
    """Auto-detect the type of data in a dataset

    Args:
        dataset: PyTorch Dataset instance

    Returns:
        Data type string: 'tabular', 'image', 'sequence', or 'unknown'
    """
    if len(dataset) == 0:
        return "unknown"

    sample = dataset[0]

    # Extract features
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        X = sample[0]
    else:
        X = sample

    # Analyze tensor shape
    if isinstance(X, torch.Tensor):
        shape = X.shape

        # 1D tensor -> tabular data
        if X.ndim == 1:
            return "tabular"

        # 3D tensor with 1 or 3 channels -> likely image (C, H, W)
        elif X.ndim == 3 and shape[0] in [1, 3, 4]:
            return "image"

        # 2D tensor -> could be sequence or flattened features
        elif X.ndim == 2:
            # If small second dimension, likely features
            if shape[1] < 50:
                return "tabular"
            else:
                return "sequence"

    # Non-tensor data
    elif isinstance(X, (list, tuple)):
        # Check if it's a sequence of numbers (tabular)
        if all(isinstance(x, (int, float)) for x in X):
            return "tabular"

    return "unknown"


def is_tabular(dataset: Dataset) -> bool:
    """Check if dataset contains tabular data

    Args:
        dataset: PyTorch Dataset instance

    Returns:
        True if tabular, False otherwise
    """
    return detect_data_type(dataset) == "tabular"


def is_image(dataset: Dataset) -> bool:
    """Check if dataset contains image data

    Args:
        dataset: PyTorch Dataset instance

    Returns:
        True if image, False otherwise
    """
    return detect_data_type(dataset) == "image"


def is_sequence(dataset: Dataset) -> bool:
    """Check if dataset contains sequence data

    Args:
        dataset: PyTorch Dataset instance

    Returns:
        True if sequence, False otherwise
    """
    return detect_data_type(dataset) == "sequence"
