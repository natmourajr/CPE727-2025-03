#!/usr/bin/env python3
"""
Download Reuters RCV1 dataset using sklearn and save to HDF5 format.
"""

from pathlib import Path
import numpy as np
import h5py
from sklearn.datasets import fetch_rcv1
from scipy.sparse import csr_matrix


def download_rcv1(output_path: Path, n_features: int = None) -> None:
    """
    Download Reuters RCV1 dataset and save to HDF5 file.

    Args:
        output_path: Path object for the directory where the HDF5 file will be saved
        n_features: Number of most frequent features to keep (default: None, keeps all features)
    """
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading Reuters RCV1 dataset...")
    # Download the dataset
    rcv1 = fetch_rcv1(subset='all', download_if_missing=True)

    print(f"Dataset downloaded successfully!")
    print(f"Number of samples: {rcv1.data.shape[0]}")
    print(f"Number of features: {rcv1.data.shape[1]}")

    # Select top n_features if specified
    data = rcv1.data
    num_features = rcv1.data.shape[1]

    if n_features is not None and n_features < num_features:
        print(f"Selecting top {n_features} most frequent features...")
        # Calculate feature frequencies (sum of occurrences across all documents)
        feature_freq = np.array(data.sum(axis=0)).flatten()
        # Get indices of top n_features
        top_feature_indices = np.argsort(feature_freq)[-n_features:][::-1]
        top_feature_indices = np.sort(top_feature_indices)

        # Filter data to keep only top features
        data = data[:, top_feature_indices]
        num_features = n_features
        print(f"Reduced to {num_features} features")

    # Convert to CSR format for efficient storage and access
    print("Converting data to CSR format...")
    data_csr = csr_matrix(data)

    # Prepare target labels as list of indices for each sample
    print("Processing target labels...")
    target_labels = []
    max_labels = 0
    for i in range(rcv1.target.shape[0]):
        labels = rcv1.target[i].nonzero()[1].tolist()
        target_labels.append(labels)
        max_labels = max(max_labels, len(labels))

    # Save to HDF5
    output_file = output_path / "rcv1_dataset.h5"
    print(f"Saving to {output_file}...")

    with h5py.File(output_file, 'w') as f:
        # Store sparse matrix in CSR format
        f.create_dataset('data', data=data_csr.data, compression='gzip')
        f.create_dataset('indices', data=data_csr.indices, compression='gzip')
        f.create_dataset('indptr', data=data_csr.indptr, compression='gzip')

        # Store shape information
        f.attrs['shape'] = data_csr.shape
        f.attrs['num_samples'] = data_csr.shape[0]
        f.attrs['num_features'] = num_features

        # Store target labels as variable-length dataset
        # Create a special dtype for variable-length integer arrays
        dt = h5py.vlen_dtype(np.dtype('int32'))
        targets_dset = f.create_dataset('targets', (len(target_labels),), dtype=dt)
        for i, labels in enumerate(target_labels):
            targets_dset[i] = np.array(labels, dtype='int32')

        # Store target names
        target_names_encoded = [name.encode('utf-8') for name in rcv1.target_names]
        f.create_dataset('target_names', data=target_names_encoded)

        # Store metadata
        f.attrs['num_target_categories'] = len(rcv1.target_names)
        f.attrs['original_num_features'] = rcv1.data.shape[1]

    print(f"Dataset saved successfully to {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")

    # Save metadata
    metadata_file = output_path / "rcv1_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Reuters RCV1 Dataset Metadata\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Number of samples: {data.shape[0]}\n")
        f.write(f"Original number of features: {rcv1.data.shape[1]}\n")
        if n_features is not None and n_features < rcv1.data.shape[1]:
            f.write(f"Selected features: {num_features} (top most frequent)\n")
        else:
            f.write(f"Number of features: {num_features}\n")
        f.write(f"Number of target categories: {len(rcv1.target_names)}\n")
        f.write(f"Target names: {', '.join(rcv1.target_names[:10])}...\n")
        f.write(f"\nDescription:\n{rcv1.DESCR}\n")

    print(f"Metadata saved to {metadata_file}")
