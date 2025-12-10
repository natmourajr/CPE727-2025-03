"""
PyTorch Dataset implementation for IARA underwater acoustic database.

Provides efficient data loading and preprocessing for deep learning experiments.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wavfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
import warnings
from .features import LOFARExtractor, MELExtractor, resample_audio


class IARADataset(Dataset):
    """
    PyTorch Dataset for IARA underwater acoustic recordings.

    Supports multiple feature extraction methods (LOFAR, MEL) and flexible
    target classification schemes.
    """

    # Vessel length thresholds (in meters)
    SMALL_THRESHOLD = 100
    MEDIUM_THRESHOLD = 200

    def __init__(self,
                 data_dir: str,
                 csv_path: str,
                 feature_extractor: str = 'mel',
                 target_type: str = 'size_class',
                 data_collection: Optional[List[str]] = None,
                 sample_rate: int = 16000,
                 max_duration: Optional[float] = None,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 256,
                 transform: Optional[Callable] = None,
                 cache_features: bool = False):
        """
        Initialize IARA dataset.

        Args:
            data_dir: Directory containing audio files
            csv_path: Path to IARA metadata CSV file
            feature_extractor: Type of features ('lofar' or 'mel')
            target_type: Classification target type:
                - 'size_class': Small/Medium/Large/Background (default)
                - 'vessel_type': Cargo/Fishing/Special Craft/etc
                - 'binary': Ship noise vs Background
            data_collection: List of data collections to use (e.g., ['A', 'B', 'C'])
                           If None, uses all collections
            sample_rate: Target sample rate for audio (default: 16000 Hz)
            max_duration: Maximum audio duration in seconds (for padding/truncation)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands (for MEL extractor)
            transform: Optional transform to apply to features
            cache_features: If True, cache extracted features in memory
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.transform = transform
        self.cache_features = cache_features
        self.feature_cache = {} if cache_features else None

        # Load metadata
        self.metadata = pd.read_csv(csv_path)

        # Filter by data collection if specified
        if data_collection is not None:
            self.metadata = self.metadata[self.metadata['Dataset'].isin(data_collection)]

        # Reset index after filtering
        self.metadata = self.metadata.reset_index(drop=True)

        # Setup feature extractor
        if feature_extractor.lower() == 'lofar':
            self.extractor = LOFARExtractor(
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length
            )
        elif feature_extractor.lower() == 'mel':
            self.extractor = MELExtractor(
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")

        # Setup target encoding
        self.target_type = target_type
        self._setup_targets()

    def _setup_targets(self):
        """Setup target labels based on target_type."""
        if self.target_type == 'size_class':
            # Classify by vessel size: Small/Medium/Large/Background
            self.metadata['target'] = self.metadata.apply(self._get_size_class, axis=1)
            self.class_names = ['Small', 'Medium', 'Large', 'Background']

        elif self.target_type == 'vessel_type':
            # Use detailed vessel type
            # Map background noise to a special category
            self.metadata['target'] = self.metadata['SHIPTYPE'].fillna('Background')
            self.class_names = sorted(self.metadata['target'].unique().tolist())

        elif self.target_type == 'binary':
            # Binary: Ship noise vs Background
            self.metadata['target'] = self.metadata['SHIPTYPE'].apply(
                lambda x: 'Ship' if pd.notna(x) else 'Background'
            )
            self.class_names = ['Background', 'Ship']

        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        # Create label encoder
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Convert targets to indices
        self.metadata['target_idx'] = self.metadata['target'].map(self.label_to_idx)

    def _get_size_class(self, row) -> str:
        """Classify vessel by size based on length."""
        if pd.isna(row['Length']):
            return 'Background'

        length = float(row['Length'])
        if length < self.SMALL_THRESHOLD:
            return 'Small'
        elif length < self.MEDIUM_THRESHOLD:
            return 'Medium'
        else:
            return 'Large'

    def _find_audio_file(self, file_id: str) -> Optional[Path]:
        """
        Find audio file by ID.

        Args:
            file_id: File ID from metadata (e.g., '0001')

        Returns:
            Path to audio file or None if not found
        """
        # Try common patterns
        patterns = [
            f"*-{file_id}.wav",
            f"*_{file_id}.wav",
            f"{file_id}.wav"
        ]

        for pattern in patterns:
            matches = list(self.data_dir.rglob(pattern))
            if matches:
                return matches[0]

        return None

    def _load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio, sample_rate)
        """
        sr, audio = wavfile.read(file_path)

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        return audio, sr

    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio: resample, pad/truncate.

        Args:
            audio: Audio signal
            sr: Original sample rate

        Returns:
            Preprocessed audio
        """
        # Resample if needed
        if sr != self.sample_rate:
            audio = resample_audio(audio, sr, self.sample_rate)

        # Pad or truncate to max_duration if specified
        if self.max_duration is not None:
            target_length = int(self.max_duration * self.sample_rate)
            current_length = len(audio)

            if current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                audio = np.pad(audio, (0, padding), mode='constant')
            elif current_length > target_length:
                # Truncate
                audio = audio[:target_length]

        return audio

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, target, metadata)
            - features: PyTorch tensor of shape [channels, freq, time]
            - target: Integer class label
            - metadata: Dictionary with sample information
        """
        # Check cache first
        if self.cache_features and idx in self.feature_cache:
            return self.feature_cache[idx]

        # Get metadata
        row = self.metadata.iloc[idx]
        file_id = row['ID']
        target = row['target_idx']

        # Load audio
        file_path = self._find_audio_file(file_id)
        if file_path is None:
            raise FileNotFoundError(f"Audio file not found for ID: {file_id}")

        audio, sr = self._load_audio(file_path)
        audio = self._preprocess_audio(audio, sr)

        # Extract features
        features = self.extractor.extract_torch(audio)

        # Apply transform if specified
        if self.transform is not None:
            features = self.transform(features)

        # Prepare metadata
        metadata = {
            'file_id': file_id,
            'dataset': row['Dataset'],
            'vessel_name': row.get('Name', 'Unknown'),
            'vessel_type': row.get('SHIPTYPE', 'Background'),
            'length': row.get('Length', 0),
            'target_label': self.idx_to_label[target]
        }

        result = (features, target, metadata)

        # Cache if enabled
        if self.cache_features:
            self.feature_cache[idx] = result

        return result

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset.

        Returns:
            Tensor of class weights (inverse frequency)
        """
        class_counts = self.metadata['target_idx'].value_counts().sort_index()
        total = len(self.metadata)
        weights = total / (len(class_counts) * class_counts.values)
        return torch.FloatTensor(weights)

    def get_split_indices(self, train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          random_state: int = 42) -> Tuple[List[int], List[int], List[int]]:
        """
        Generate train/val/test split indices with stratification.

        Args:
            train_ratio: Proportion of training data
            val_ratio: Proportion of validation data
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        from sklearn.model_selection import train_test_split

        # Get all indices
        indices = np.arange(len(self))
        targets = self.metadata['target_idx'].values

        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=1-train_ratio,
            stratify=targets,
            random_state=random_state
        )

        # Second split: val vs test
        val_size = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1-val_size,
            stratify=targets[temp_idx],
            random_state=random_state
        )

        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    def get_info(self) -> Dict:
        """Get dataset statistics and information."""
        info = {
            'total_samples': len(self),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_distribution': self.metadata['target'].value_counts().to_dict(),
            'data_collections': self.metadata['Dataset'].unique().tolist(),
            'feature_extractor': type(self.extractor).__name__,
            'sample_rate': self.sample_rate,
            'target_type': self.target_type
        }
        return info


def create_iara_dataloaders(
    data_dir: str,
    csv_path: str,
    batch_size: int = 32,
    feature_extractor: str = 'mel',
    target_type: str = 'size_class',
    data_collection: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    random_state: int = 42,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, IARADataset]:
    """
    Create train/val/test DataLoaders for IARA dataset.

    Args:
        data_dir: Directory containing audio files
        csv_path: Path to metadata CSV
        batch_size: Batch size for dataloaders
        feature_extractor: 'lofar' or 'mel'
        target_type: Classification target type
        data_collection: List of data collections to use
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        num_workers: Number of worker processes for data loading
        random_state: Random seed
        **dataset_kwargs: Additional arguments for IARADataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    # Create full dataset
    dataset = IARADataset(
        data_dir=data_dir,
        csv_path=csv_path,
        feature_extractor=feature_extractor,
        target_type=target_type,
        data_collection=data_collection,
        **dataset_kwargs
    )

    # Get split indices
    train_idx, val_idx, test_idx = dataset.get_split_indices(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_state=random_state
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, dataset
