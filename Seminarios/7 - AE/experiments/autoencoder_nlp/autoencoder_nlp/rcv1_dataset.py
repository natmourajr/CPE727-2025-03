#!/usr/bin/env python3
"""
PyTorch Dataset class for Reuters RCV1 dataset stored in HDF5 format.
"""

from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix


class RCV1Dataset(Dataset):
    """
    PyTorch Dataset for Reuters RCV1 dataset stored in HDF5 format.

    The dataset returns dense tensors for features and lists of label indices for targets.
    """

    def __init__(self, h5_path: Path):
        """
        Initialize the dataset.

        Args:
            h5_path: Path to the HDF5 file containing the RCV1 dataset
        """
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')

        # Load sparse matrix structure into memory
        self.data = self.h5_file['data'][:]
        self.indices = self.h5_file['indices'][:]
        self.indptr = self.h5_file['indptr'][:]

        # Get shape information
        self.num_samples = self.h5_file.attrs['num_samples']
        self.num_features = self.h5_file.attrs['num_features']
        self.shape = tuple(self.h5_file.attrs['shape'])

        # Reconstruct the sparse matrix (kept in sparse format to save memory)
        print("Loading sparse matrix structure...")
        self.sparse_matrix = csr_matrix(
            (self.data, self.indices, self.indptr),
            shape=self.shape
        )

        print(f"Dataset loaded: {self.num_samples} samples, {self.num_features} features")
        print(f"Sparse matrix memory: {(self.data.nbytes + self.indices.nbytes + self.indptr.nbytes) / (1024**2):.2f} MB")
        print("Features will be densified and L2-normalized on-the-fly during loading")

        # Keep targets in memory (they're small)
        self.targets = [np.array(self.h5_file['targets'][i]) for i in range(self.num_samples)]

        # Load target names
        target_names_encoded = self.h5_file['target_names'][:]
        self.target_names = [name.decode('utf-8') for name in target_names_encoded]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            tuple: (features, targets) where:
                - features: Dense tensor of shape (num_features,)
                - targets: List of label indices
        """
        # Extract sparse row and convert to dense
        sparse_row = self.sparse_matrix.getrow(idx)
        dense_features = np.asarray(sparse_row.todense()).squeeze()

        # L2 normalize the features
        norm = np.linalg.norm(dense_features, ord=2)
        if norm > 1e-12:
            dense_features = dense_features / norm

        # Convert to PyTorch tensor
        features = torch.from_numpy(dense_features).float()

        # Get target labels as list
        target_labels = self.targets[idx].tolist()

        return features, target_labels

    def close(self):
        """Close the HDF5 file."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        """Ensure the HDF5 file is closed when the object is destroyed."""
        self.close()


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-length target lists.

    Args:
        batch: List of (features, targets) tuples

    Returns:
        tuple: (features_batch, targets_batch) where:
            - features_batch: Tensor of shape (batch_size, num_features)
            - targets_batch: List of lists of label indices
    """
    features = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return features, targets


# Example usage
if __name__ == "__main__":
    # Load the dataset
    dataset_path = Path("./data/rcv1_dataset.h5")

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please run download_dataset.py first to download the data.")
    else:
        # Create dataset
        dataset = RCV1Dataset(dataset_path)

        print(f"Dataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of target categories: {len(dataset.target_names)}")

        # Get a sample
        features, targets = dataset[0]
        print(f"\nFirst sample:")
        print(f"Features shape: {features.shape}")
        print(f"Features (non-zero): {features.nonzero().shape[0]}")
        print(f"Target labels: {targets}")
        print(f"Target names: {[dataset.target_names[i] for i in targets]}")

        # Create DataLoader
        batch_size = 32
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 for HDF5 files (they don't work well with multiprocessing)
        )

        # Iterate through a few batches
        print(f"\nDataLoader example (batch_size={batch_size}):")
        for i, (batch_features, batch_targets) in enumerate(dataloader):
            print(f"Batch {i+1}:")
            print(f"  Features shape: {batch_features.shape}")
            print(f"  Number of samples: {len(batch_targets)}")
            print(f"  First sample targets: {batch_targets[0]}")

            if i >= 2:  # Show only first 3 batches
                break

        # Clean up
        dataset.close()
