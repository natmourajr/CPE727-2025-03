"""
Production training script with smart caching and MLflow integration.

Features:
- Checks for existing preprocessed features (avoids recomputation)
- Caches hyperparameter tuning results
- Resumes from checkpoints if available
- MLflow experiment tracking
- No JupyterLab dependency

Usage:
    # Basic usage (default MEL features, H DC only)
    python scripts/train_production.py

    # Full dataset with all DCs
    python scripts/train_production.py --all-dcs

    # Use LOFAR features
    python scripts/train_production.py --feature-type lofar

    # Force recompute preprocessing
    python scripts/train_production.py --force-preprocess

    # Custom cache directory
    python scripts/train_production.py --cache-dir ./custom_cache
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import mlflow
from src.data.metadata import MetadataManager
from src.data.preprocessing import AudioPreprocessor, SpectrogramType
from src.data.dataset import IARAAudioDataset
from src.models.cnn import get_iara_cnn
from src.training.trainer import CrossValidationTrainer


class ArtifactCache:
    """Manages caching of preprocessing and training artifacts."""

    def __init__(self, cache_dir: Path):
        """
        Initialize artifact cache.

        Args:
            cache_dir: Directory to store cached artifacts
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_config_hash(self, config: Dict) -> str:
        """Compute hash of configuration for cache key."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def check_preprocessing_cache(self, config: Dict) -> Optional[Path]:
        """
        Check if preprocessing artifacts exist for given config.

        Args:
            config: Preprocessing configuration dict

        Returns:
            Path to cached features if exists, None otherwise
        """
        cache_key = self._compute_config_hash(config)
        cache_entry = f"preprocess_{cache_key}"

        if cache_entry in self.metadata:
            feature_dir = Path(self.metadata[cache_entry]['path'])
            if feature_dir.exists():
                print(f"✓ Found cached preprocessing artifacts: {feature_dir}")
                return feature_dir

        return None

    def save_preprocessing_cache(self, config: Dict, feature_dir: Path):
        """
        Save preprocessing cache metadata.

        Args:
            config: Preprocessing configuration dict
            feature_dir: Directory containing cached features
        """
        cache_key = self._compute_config_hash(config)
        cache_entry = f"preprocess_{cache_key}"

        self.metadata[cache_entry] = {
            'config': config,
            'path': str(feature_dir),
            'type': 'preprocessing'
        }
        self._save_metadata()
        print(f"✓ Saved preprocessing cache: {feature_dir}")

    def check_hyperparameter_cache(self, model_name: str) -> Optional[Dict]:
        """
        Check if hyperparameter tuning results exist.

        Args:
            model_name: Name of the model

        Returns:
            Best hyperparameters if cached, None otherwise
        """
        cache_entry = f"hparams_{model_name}"

        if cache_entry in self.metadata:
            hparams = self.metadata[cache_entry]['hparams']
            print(f"✓ Found cached hyperparameters for {model_name}:")
            for k, v in hparams.items():
                print(f"  - {k}: {v}")
            return hparams

        return None

    def save_hyperparameter_cache(self, model_name: str, hparams: Dict):
        """
        Save hyperparameter tuning results.

        Args:
            model_name: Name of the model
            hparams: Best hyperparameters found
        """
        cache_entry = f"hparams_{model_name}"

        self.metadata[cache_entry] = {
            'model': model_name,
            'hparams': hparams,
            'type': 'hyperparameters'
        }
        self._save_metadata()
        print(f"✓ Saved hyperparameter cache for {model_name}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Production training script with smart caching"
    )

    # Data arguments
    parser.add_argument(
        '--data-root',
        type=Path,
        default=project_root / "data" / "downloaded_content",
        help="Root directory containing IARA data"
    )
    parser.add_argument(
        '--all-dcs',
        action='store_true',
        help="Use all DCs (A-E). Default: only H DC for testing"
    )

    # Preprocessing arguments
    parser.add_argument(
        '--feature-type',
        type=str,
        choices=['mel', 'lofar'],
        default='mel',
        help="Feature type to extract (mel or lofar)"
    )
    parser.add_argument(
        '--force-preprocess',
        action='store_true',
        help="Force recompute preprocessing even if cached"
    )

    # Training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=50,
        help="Maximum epochs per fold"
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-3,
        help="Weight decay"
    )

    # Cache arguments
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=project_root / "experiments" / "cache",
        help="Directory for caching artifacts"
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=project_root / "experiments" / "production",
        help="Output directory for results"
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default="iara_production",
        help="MLflow experiment name"
    )

    return parser.parse_args()


def setup_mlflow(output_dir: Path, experiment_name: str):
    """
    Setup MLflow tracking.

    Args:
        output_dir: Output directory
        experiment_name: Experiment name
    """
    mlruns_dir = output_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"file://{mlruns_dir.absolute()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"✓ MLflow tracking URI: {tracking_uri}")
    print(f"✓ MLflow experiment: {experiment_name}")


def load_or_create_dataset(
    data_root: Path,
    cache: ArtifactCache,
    feature_type: str,
    all_dcs: bool,
    force_preprocess: bool,
) -> IARAAudioDataset:
    """
    Load dataset with cached preprocessing if available.

    Args:
        data_root: Root directory containing data
        cache: Artifact cache manager
        feature_type: Feature type (mel or lofar)
        all_dcs: Whether to use all DCs
        force_preprocess: Force recompute preprocessing

    Returns:
        IARA dataset with preprocessed features
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING")
    print("=" * 70)

    # Setup paths
    csv_path = project_root / "IARA" / "src" / "iara" / "dataset_info" / "iara.csv"
    xlsx_path = data_root / "iara.xlsx"

    # Preprocessing configuration
    preprocess_config = {
        'target_sr': 16000,
        'n_fft': 1024,
        'hop_length': 1024,
        'n_mels': 128,
        'averaging_windows': 8,
        'feature_type': feature_type,
        'use_windows': True,
        'window_size': 32,
        'dcs': None if all_dcs else ['H'],
    }

    # Check cache
    cached_features = None
    if not force_preprocess:
        cached_features = cache.check_preprocessing_cache(preprocess_config)

    # Initialize metadata manager
    print("\n1. Loading metadata...")
    manager = MetadataManager(
        csv_path=csv_path if csv_path.exists() else None,
        xlsx_path=xlsx_path if xlsx_path.exists() else None,
        data_root=data_root
    )
    df = manager.load_metadata()
    print(f"   ✓ Loaded {len(df)} recordings")

    # Initialize preprocessor
    print("\n2. Setting up preprocessor...")
    preprocessor = AudioPreprocessor(
        target_sr=preprocess_config['target_sr'],
        n_fft=preprocess_config['n_fft'],
        hop_length=preprocess_config['hop_length'],
        n_mels=preprocess_config['n_mels'],
        averaging_windows=preprocess_config['averaging_windows'],
    )
    print(f"   ✓ n_fft={preprocessor.n_fft}, hop_length={preprocessor.hop_length}")
    print(f"   ✓ n_mels={preprocessor.n_mels}, averaging_windows={preprocessor.averaging_windows}")

    # Create dataset
    print("\n3. Creating dataset...")
    spec_type = SpectrogramType.MEL if feature_type == 'mel' else SpectrogramType.LOFAR

    # Disable in-memory caching for large datasets to avoid OOM
    # Features are still computed on-the-fly efficiently
    cache_features = False

    dataset = IARAAudioDataset(
        data_root=data_root,
        metadata_manager=manager,
        preprocessor=preprocessor,
        feature_type=spec_type,
        use_windows=preprocess_config['use_windows'],
        window_size=preprocess_config['window_size'],
        dcs=preprocess_config['dcs'],
        cache_features=cache_features,
    )

    print(f"   ✓ Dataset created with {len(dataset)} samples")
    print(f"   ✓ Class distribution:")
    for cls, count in dataset.metadata_df['Class'].value_counts().items():
        print(f"      - {cls}: {count}")

    # Save cache metadata if preprocessing was done
    if cache_features and cached_features is None:
        feature_dir = cache.cache_dir / f"features_{feature_type}"
        cache.save_preprocessing_cache(preprocess_config, feature_dir)

    return dataset, manager


def get_hyperparameters(
    cache: ArtifactCache,
    model_name: str,
    default_hparams: Dict,
) -> Dict:
    """
    Get hyperparameters from cache or use defaults.

    Args:
        cache: Artifact cache manager
        model_name: Name of the model
        default_hparams: Default hyperparameters

    Returns:
        Hyperparameters to use
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETERS")
    print("=" * 70)

    # Check cache
    cached_hparams = cache.check_hyperparameter_cache(model_name)

    if cached_hparams is not None:
        return cached_hparams

    print(f"\nℹ No cached hyperparameters found for {model_name}")
    print("Using default hyperparameters:")
    for k, v in default_hparams.items():
        print(f"  - {k}: {v}")

    # Save defaults to cache for future runs
    cache.save_hyperparameter_cache(model_name, default_hparams)

    return default_hparams


def train_model(
    dataset: IARAAudioDataset,
    manager: MetadataManager,
    hparams: Dict,
    output_dir: Path,
    experiment_name: str,
) -> Dict:
    """
    Train model with cross-validation.

    Args:
        dataset: IARA dataset
        manager: Metadata manager
        hparams: Hyperparameters
        output_dir: Output directory
        experiment_name: MLflow experiment name

    Returns:
        Training results
    """
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    # Model factory function
    def create_model():
        return get_iara_cnn(
            num_classes=manager.num_classes,
            config="mel",  # TODO: make configurable
            pretrained=False,
        )

    # Initialize trainer with MLflow
    print("\n1. Setting up 5x2 cross-validation trainer...")
    trainer = CrossValidationTrainer(
        dataset=dataset,
        model_fn=create_model,
        num_classes=manager.num_classes,
        class_names=manager.class_names,
        batch_size=hparams['batch_size'],
        max_epochs=hparams['max_epochs'],
        learning_rate=hparams['learning_rate'],
        weight_decay=hparams['weight_decay'],
        early_stopping_patience=hparams.get('early_stopping_patience', 10),
        output_dir=output_dir,
        use_class_weights=True,
        experiment_name=experiment_name,
        use_mlflow=True,
    )

    # Run cross-validation
    print("\n2. Starting 5x2 cross-validation...")
    results = trainer.run_5x2_cv()

    return results


def main():
    """Main training pipeline."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("IARA PRODUCTION TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data root:       {args.data_root}")
    print(f"  Feature type:    {args.feature_type}")
    print(f"  All DCs:         {args.all_dcs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Max epochs:      {args.max_epochs}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Weight decay:    {args.weight_decay}")
    print(f"  Cache dir:       {args.cache_dir}")
    print(f"  Output dir:      {args.output_dir}")

    # Initialize artifact cache
    cache = ArtifactCache(args.cache_dir)

    # Setup MLflow
    setup_mlflow(args.output_dir, args.experiment_name)

    # Step 1: Load or create preprocessed dataset
    dataset, manager = load_or_create_dataset(
        data_root=args.data_root,
        cache=cache,
        feature_type=args.feature_type,
        all_dcs=args.all_dcs,
        force_preprocess=args.force_preprocess,
    )

    # Step 2: Get hyperparameters (from cache or defaults)
    default_hparams = {
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': 10,
    }
    hparams = get_hyperparameters(
        cache=cache,
        model_name='baseline_cnn',
        default_hparams=default_hparams,
    )

    # Step 3: Train model
    results = train_model(
        dataset=dataset,
        manager=manager,
        hparams=hparams,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    # Print results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    summary = results['summary']
    print("\nFinal Results (Mean ± Std):")
    print(f"  SP:                 {summary['sp'][0]:.4f} ± {summary['sp'][1]:.4f}")
    print(f"  Balanced Accuracy:  {summary['balanced_accuracy'][0]:.4f} ± {summary['balanced_accuracy'][1]:.4f}")
    print(f"  F1-Score (macro):   {summary['f1_macro'][0]:.4f} ± {summary['f1_macro'][1]:.4f}")

    print("\n✓ Results saved to:", args.output_dir)
    print("\nView results in MLflow UI:")
    print(f"  mlflow ui --backend-store-uri file://{args.output_dir / 'mlruns'}")
    print(f"  http://localhost:5000")


if __name__ == "__main__":
    main()
