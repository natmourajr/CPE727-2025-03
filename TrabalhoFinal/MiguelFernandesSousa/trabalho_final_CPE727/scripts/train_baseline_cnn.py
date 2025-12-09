"""
Example training script for IARA baseline CNN model.

Usage:
    python train_baseline_cnn.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.data.metadata import MetadataManager
from src.data.preprocessing import AudioPreprocessor, SpectrogramType
from src.data.dataset import IARAAudioDataset
from src.models.cnn import get_iara_cnn
from src.training.trainer import CrossValidationTrainer


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("IARA Baseline CNN Training")
    print("=" * 70)
    
    # Setup paths
    data_root = project_root / "data" / "downloaded_content"
    csv_path = project_root / "IARA" / "src" / "iara" / "dataset_info" / "iara.csv"
    xlsx_path = data_root / "iara.xlsx"
    
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
    print("\n2. Setting up data pipeline...")
    preprocessor = AudioPreprocessor(
        target_sr=16000,
        n_fft=1024,
        hop_length=1024,
        n_mels=128,
        averaging_windows=8,
    )
    
    # Create dataset (using only H DC for faster testing)
    # For full training, use dcs=None or dcs=["A", "B", "C", "D", "E"]
    dataset = IARAAudioDataset(
        data_root=data_root,
        metadata_manager=manager,
        preprocessor=preprocessor,
        feature_type=SpectrogramType.MEL,
        use_windows=True,  # For CNN
        window_size=32,
        dcs=["H"],  # For testing with H folder only
        cache_features=False,  # Set True for faster training if RAM allows
    )
    
    print(f"   ✓ Created dataset with {len(dataset)} samples")
    print(f"   ✓ Class distribution: {dataset.metadata_df['Class'].value_counts().to_dict()}")
    
    # Model factory function
    def create_model():
        return get_iara_cnn(
            num_classes=manager.num_classes,
            config="mel",
            pretrained=False,
        )
    
    # Initialize trainer
    print("\n3. Setting up 5x2 cross-validation trainer...")
    trainer = CrossValidationTrainer(
        dataset=dataset,
        model_fn=create_model,
        num_classes=manager.num_classes,
        class_names=manager.class_names,
        batch_size=32,
        max_epochs=50,  # Reduced for testing
        learning_rate=1e-4,
        weight_decay=1e-3,
        early_stopping_patience=10,
        output_dir=project_root / "experiments" / "baseline_cnn",
        use_class_weights=True,
    )
    
    # Run cross-validation
    print("\n4. Starting training...")
    results = trainer.run_5x2_cv()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # Print final results
    summary = results['summary']
    print("\nFinal Results (Mean ± Std):")
    print(f"  SP:                 {summary['sp'][0]:.4f} ± {summary['sp'][1]:.4f}")
    print(f"  Balanced Accuracy:  {summary['balanced_accuracy'][0]:.4f} ± {summary['balanced_accuracy'][1]:.4f}")
    print(f"  F1-Score (macro):   {summary['f1_macro'][0]:.4f} ± {summary['f1_macro'][1]:.4f}")
    
    print("\n✓ Results saved to:", trainer.output_dir)


if __name__ == "__main__":
    main()
