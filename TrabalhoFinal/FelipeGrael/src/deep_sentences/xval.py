import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import KFold
from typing import Optional


def create_kfold_splits(
    n_samples: int,
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = 202512,
    output_path: str = "xval_splits.h5",
):
    """
    Create k-fold cross-validation splits and save indices to HDF5 file.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset
    n_folds : int, default=5
        Number of folds for cross-validation
    shuffle : bool, default=True
        Whether to shuffle data before splitting
    random_state : int or None, default=202512
        Random seed for reproducibility
    output_path : str, default="xval_splits.h5"
        Path to save the HDF5 file with split indices

    Returns
    -------
    str
        Path to the created HDF5 file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.arange(n_samples)

    kfold = KFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=random_state,
    )
    splits = kfold.split(indices)

    with h5py.File(output_path, "w") as f:
        f.attrs["n_samples"] = n_samples
        f.attrs["n_folds"] = n_folds
        f.attrs["shuffle"] = shuffle
        f.attrs["random_state"] = random_state if random_state is not None else -1

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_group = f.create_group(f"fold_{fold_idx}")
            fold_group.create_dataset("train", data=train_idx, compression="gzip")
            fold_group.create_dataset("validation", data=val_idx, compression="gzip")

    print(f"Created {n_folds}-fold cross-validation splits")
    print(f"Saved to: {output_path.absolute()}")
    print(f"Total samples: {n_samples}")
    print(f"Samples per fold (approx): train={len(train_idx)}, validation={len(val_idx)}")

    return str(output_path.absolute())


def load_split(file_path: str, fold: int):
    """
    Load train/validation indices from HDF5 file for a specific fold.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing splits
    fold : int
        Fold number to load

    Returns
    -------
    tuple
        (train_indices, validation_indices) as numpy arrays
    """
    with h5py.File(file_path, "r") as f:
        fold_group = f[f"fold_{fold}"]
        train_idx = fold_group["train"][:]
        val_idx = fold_group["validation"][:]

    return train_idx, val_idx


def get_split_info(file_path: str):
    """
    Get information about saved k-fold splits.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing splits

    Returns
    -------
    dict
        Dictionary containing split metadata
    """
    with h5py.File(file_path, "r") as f:
        info = dict(f.attrs)
        info["folds"] = list(f.keys())

    return info
