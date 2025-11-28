#!/usr/bin/env python3
"""
Evaluation script for the Autoencoder model.
Encodes the RCV1 dataset to latent space representations and saves them.
"""

from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import Autoencoder
from .rcv1_dataset import RCV1Dataset, collate_fn


def encode_dataset(
    dataset_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> None:
    """
    Encode the RCV1 dataset to latent space and save results.

    Args:
        dataset_path: Path to the RCV1 HDF5 dataset file
        checkpoint_path: Path to the trained model checkpoint (.ckpt file)
        output_path: Path where the encoded latent representations will be saved
        batch_size: Batch size for encoding (larger = faster, uses more memory)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the trained model from checkpoint
    print(f"\nLoading model from checkpoint: {checkpoint_path}")
    model = Autoencoder.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    print(f"Model architecture:")
    print(f"  Dimensions: {model.dimensions}")
    print(f"  Activation: {model.activation}")
    print(f"  Latent activation: {model.latent_activation}")
    print(f"  Latent dimension: {model.dimensions[-1]}")

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = RCV1Dataset(dataset_path)
    print(f"Total samples: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep original order
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Encode all samples
    print(f"\nEncoding dataset to latent space...")
    latent_representations = []
    all_targets = []

    with torch.no_grad():
        for batch_features, batch_targets in tqdm(dataloader, desc="Encoding batches"):
            # Move to device
            batch_features = batch_features.to(device)

            # Encode to latent space
            latent = model.encode(batch_features)

            # Move back to CPU and store
            latent_representations.append(latent.cpu().numpy())
            all_targets.extend(batch_targets)

    # Concatenate all batches
    print("\nConcatenating results...")
    latent_representations = np.vstack(latent_representations)
    print(f"Latent representations shape: {latent_representations.shape}")

    # Save to HDF5 file
    print(f"\nSaving results to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Save latent representations
        f.create_dataset(
            'latent_representations',
            data=latent_representations,
            compression='gzip',
            compression_opts=9
        )

        # Save targets as variable-length datasets
        # Convert list of lists to ragged array
        dt = h5py.special_dtype(vlen=np.int32)
        target_dataset = f.create_dataset('targets', (len(all_targets),), dtype=dt)
        for i, target_list in enumerate(all_targets):
            target_dataset[i] = np.array(target_list, dtype=np.int32)

        # Save target names
        target_names_encoded = [name.encode('utf-8') for name in dataset.target_names]
        f.create_dataset('target_names', data=target_names_encoded)

        # Save metadata
        f.attrs['num_samples'] = len(dataset)
        f.attrs['latent_dim'] = latent_representations.shape[1]
        f.attrs['original_features'] = dataset.num_features
        f.attrs['checkpoint_path'] = str(checkpoint_path)
        f.attrs['dataset_path'] = str(dataset_path)
        f.attrs['model_dimensions'] = model.dimensions
        f.attrs['activation'] = model.activation
        f.attrs['latent_activation'] = model.latent_activation

    print("\n" + "=" * 80)
    print("Encoding completed successfully!")
    print(f"Latent representations shape: {latent_representations.shape}")
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    # Compute and display statistics
    print("\nLatent Space Statistics:")
    print(f"  Mean: {latent_representations.mean():.6f}")
    print(f"  Std:  {latent_representations.std():.6f}")
    print(f"  Min:  {latent_representations.min():.6f}")
    print(f"  Max:  {latent_representations.max():.6f}")

    # Compute sparsity (percentage of near-zero values)
    sparsity = np.mean(np.abs(latent_representations) < 1e-6) * 100
    print(f"  Sparsity (|x| < 1e-6): {sparsity:.2f}%")

    # Clean up
    dataset.close()


def encode_with_reconstruction(
    dataset_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 256,
    device: Optional[str] = None,
    compute_reconstruction_error: bool = True,
) -> None:
    """
    Encode dataset and optionally compute reconstruction errors.

    Args:
        dataset_path: Path to the RCV1 HDF5 dataset file
        checkpoint_path: Path to the trained model checkpoint
        output_path: Path where results will be saved
        batch_size: Batch size for encoding
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        compute_reconstruction_error: Whether to compute reconstruction errors
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from checkpoint: {checkpoint_path}")
    model = Autoencoder.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = RCV1Dataset(dataset_path)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Encode and optionally reconstruct
    print(f"\nProcessing dataset...")
    latent_representations = []
    reconstruction_errors = [] if compute_reconstruction_error else None
    all_targets = []

    with torch.no_grad():
        for batch_features, batch_targets in tqdm(dataloader, desc="Processing"):
            batch_features = batch_features.to(device)

            # Encode
            latent = model.encode(batch_features)
            latent_representations.append(latent.cpu().numpy())

            # Compute reconstruction error if requested
            if compute_reconstruction_error:
                reconstructed = model(batch_features)
                mse = torch.mean((batch_features - reconstructed) ** 2, dim=1)
                reconstruction_errors.append(mse.cpu().numpy())

            all_targets.extend(batch_targets)

    # Concatenate results
    latent_representations = np.vstack(latent_representations)
    if reconstruction_errors:
        reconstruction_errors = np.concatenate(reconstruction_errors)

    # Save results
    print(f"\nSaving results to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Save latent representations
        f.create_dataset(
            'latent_representations',
            data=latent_representations,
            compression='gzip',
            compression_opts=9
        )

        # Save reconstruction errors if computed
        if reconstruction_errors is not None:
            f.create_dataset(
                'reconstruction_errors',
                data=reconstruction_errors,
                compression='gzip',
                compression_opts=9
            )

        # Save targets
        dt = h5py.special_dtype(vlen=np.int32)
        target_dataset = f.create_dataset('targets', (len(all_targets),), dtype=dt)
        for i, target_list in enumerate(all_targets):
            target_dataset[i] = np.array(target_list, dtype=np.int32)

        # Save target names
        target_names_encoded = [name.encode('utf-8') for name in dataset.target_names]
        f.create_dataset('target_names', data=target_names_encoded)

        # Save metadata
        f.attrs['num_samples'] = len(dataset)
        f.attrs['latent_dim'] = latent_representations.shape[1]
        f.attrs['original_features'] = dataset.num_features
        f.attrs['checkpoint_path'] = str(checkpoint_path)
        f.attrs['dataset_path'] = str(dataset_path)

    print("\n" + "=" * 80)
    print("Processing completed!")
    print(f"Results saved to: {output_path}")

    if reconstruction_errors is not None:
        print(f"\nReconstruction Error Statistics:")
        print(f"  Mean MSE: {reconstruction_errors.mean():.6f}")
        print(f"  Std MSE:  {reconstruction_errors.std():.6f}")
        print(f"  Min MSE:  {reconstruction_errors.min():.6f}")
        print(f"  Max MSE:  {reconstruction_errors.max():.6f}")

    print("=" * 80)

    # Clean up
    dataset.close()


