"""
Hyperparameter tuning script for IARA models.

Performs grid search over hyperparameters and logs results to MLflow.

Usage:
    # Quick search with H DC only
    python scripts/tune_hyperparameters.py

    # Full search with all DCs
    python scripts/tune_hyperparameters.py --all-dcs

    # Custom search space
    python scripts/tune_hyperparameters.py --learning-rates 1e-3 1e-4 1e-5 --batch-sizes 16 32 64
"""

import sys
import argparse
import itertools
from pathlib import Path
from typing import Dict, List

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with grid search"
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
        help="Use all DCs (A-E). Default: A, B, C, D for multi-class"
    )

    # Feature arguments
    parser.add_argument(
        '--feature-type',
        type=str,
        choices=['mel', 'lofar'],
        default='mel',
        help="Feature type to extract"
    )

    # Hyperparameter search space
    parser.add_argument(
        '--learning-rates',
        nargs='+',
        type=float,
        default=[1e-4, 5e-4, 1e-3],
        help="Learning rates to try"
    )
    parser.add_argument(
        '--batch-sizes',
        nargs='+',
        type=int,
        default=[32, 64],
        help="Batch sizes to try"
    )
    parser.add_argument(
        '--weight-decays',
        nargs='+',
        type=float,
        default=[1e-3, 5e-3],
        help="Weight decay values to try"
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=50,
        help="Maximum epochs per trial"
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=project_root / "experiments" / "hyperparameter_tuning",
        help="Output directory for results"
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default="iara_hyperparameter_tuning",
        help="MLflow experiment name"
    )

    return parser.parse_args()


def setup_mlflow(output_dir: Path, experiment_name: str):
    """Setup MLflow tracking."""
    mlruns_dir = output_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"file://{mlruns_dir.absolute()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"✓ MLflow tracking URI: {tracking_uri}")
    print(f"✓ MLflow experiment: {experiment_name}")


def create_dataset(args):
    """Create dataset for tuning."""
    print("\n" + "=" * 70)
    print("DATASET PREPARATION")
    print("=" * 70)

    # Setup paths
    csv_path = project_root / "IARA" / "src" / "iara" / "dataset_info" / "iara.csv"
    xlsx_path = args.data_root / "iara.xlsx"

    # Initialize metadata manager
    print("\n1. Loading metadata...")
    manager = MetadataManager(
        csv_path=csv_path if csv_path.exists() else None,
        xlsx_path=xlsx_path if xlsx_path.exists() else None,
        data_root=args.data_root
    )
    df = manager.load_metadata()
    print(f"   ✓ Loaded {len(df)} recordings")

    # Initialize preprocessor
    print("\n2. Setting up preprocessor...")
    preprocessor = AudioPreprocessor(
        target_sr=16000,
        n_fft=1024,
        hop_length=1024,
        n_mels=128,
        averaging_windows=8,
    )

    # Create dataset
    # Use A, B, C, D for multi-class (avoid E and H which are background-only)
    if args.all_dcs:
        dcs = None  # All DCs
    else:
        dcs = ["A", "B", "C", "D"]  # Multi-class training

    print(f"\n3. Creating dataset (DCs: {dcs or 'ALL'})...")
    spec_type = SpectrogramType.MEL if args.feature_type == 'mel' else SpectrogramType.LOFAR

    dataset = IARAAudioDataset(
        data_root=args.data_root,
        metadata_manager=manager,
        preprocessor=preprocessor,
        feature_type=spec_type,
        use_windows=True,
        window_size=32,
        dcs=dcs,
        cache_features=False,  # Disable to avoid OOM on large datasets
    )

    print(f"   ✓ Dataset created with {len(dataset)} samples")
    print(f"   ✓ Class distribution:")
    for cls, count in dataset.metadata_df['Class'].value_counts().items():
        print(f"      - {cls}: {count}")

    return dataset, manager


def run_grid_search(dataset, manager, args):
    """Run grid search over hyperparameters."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 70)

    # Generate all combinations
    param_grid = list(itertools.product(
        args.learning_rates,
        args.batch_sizes,
        args.weight_decays
    ))

    print(f"\nSearching {len(param_grid)} parameter combinations:")
    print(f"  Learning rates: {args.learning_rates}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Weight decays: {args.weight_decays}")
    print(f"  Max epochs: {args.max_epochs}")

    best_score = 0.0
    best_params = None
    results = []

    for i, (lr, batch_size, weight_decay) in enumerate(param_grid, 1):
        print(f"\n{'=' * 70}")
        print(f"Trial {i}/{len(param_grid)}")
        print(f"{'=' * 70}")
        print(f"  lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")

        # Model factory
        def create_model():
            return get_iara_cnn(
                num_classes=manager.num_classes,
                config="mel",
                pretrained=False,
            )

        # Initialize trainer
        trainer = CrossValidationTrainer(
            dataset=dataset,
            model_fn=create_model,
            num_classes=manager.num_classes,
            class_names=manager.class_names,
            batch_size=batch_size,
            max_epochs=args.max_epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
            early_stopping_patience=10,
            output_dir=args.output_dir / f"trial_{i}",
            use_class_weights=True,
            experiment_name=args.experiment_name,
            use_mlflow=True,
        )

        # Run 5x2 CV
        try:
            cv_results = trainer.run_5x2_cv()

            # Get mean balanced accuracy
            mean_ba = cv_results['summary']['balanced_accuracy'][0]

            print(f"\n  Result: Balanced Accuracy = {mean_ba:.4f}")

            results.append({
                'learning_rate': lr,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'balanced_accuracy': mean_ba,
                'sp': cv_results['summary']['sp'][0],
                'f1_macro': cv_results['summary']['f1_macro'][0],
            })

            # Track best
            if mean_ba > best_score:
                best_score = mean_ba
                best_params = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'weight_decay': weight_decay,
                }
                print(f"  ✓ New best score!")

        except Exception as e:
            print(f"  ✗ Trial failed: {e}")
            continue

    return results, best_params, best_score


def main():
    """Main hyperparameter tuning pipeline."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("IARA HYPERPARAMETER TUNING")
    print("=" * 70)

    # Setup MLflow
    setup_mlflow(args.output_dir, args.experiment_name)

    # Create dataset
    dataset, manager = create_dataset(args)

    # Run grid search
    results, best_params, best_score = run_grid_search(dataset, manager, args)

    # Print summary
    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)

    if best_params is None:
        print("\n✗ All trials failed. Check errors above.")
        print(f"\nResults directory: {args.output_dir}")
        return

    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest balanced accuracy: {best_score:.4f}")

    print(f"\n✓ Results saved to: {args.output_dir}")
    print(f"\nView results in MLflow UI:")
    print(f"  mlflow ui --backend-store-uri file://{args.output_dir / 'mlruns'}")
    print(f"  http://localhost:5000")


if __name__ == "__main__":
    main()
