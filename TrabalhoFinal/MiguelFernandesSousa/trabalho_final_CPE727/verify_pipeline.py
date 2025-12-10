#!/usr/bin/env python3
"""
Quick verification script for IARA ML Pipeline.

Tests all major components with the H folder data.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 70)
    print("1. Testing Module Imports")
    print("=" * 70)
    
    try:
        from src.data.metadata import MetadataManager
        from src.data.preprocessing import AudioPreprocessor, SpectrogramType
        from src.data.dataset import IARAAudioDataset
        from src.models.cnn import IARACNN, get_iara_cnn
        from src.models.mlp import IARAMLP, get_iara_mlp
        from src.evaluation import compute_metrics, sum_product_index, MetricsTracker
        from src.training.trainer import IARAClassifier, CrossValidationTrainer
        
        print("✓ All core modules imported successfully")
        
        # Try optional imports
        try:
            from src.models.resnet import get_resnet18
            from src.models.efficientnet import get_efficientnet_b0
            from src.models.convnext import get_convnext_tiny
            print("✓ Advanced models (timm) available")
        except ImportError as e:
            print(f"⚠️  Advanced models not available (install timm): {e}")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading with H folder."""
    print("\n" + "=" * 70)
    print("2. Testing Data Loading (H Folder)")
    print("=" * 70)
    
    try:
        from src.data.metadata import MetadataManager
        
        data_root = project_root / "data" / "downloaded_content"
        csv_path = project_root / "IARA" / "src" / "iara" / "dataset_info" / "iara.csv"
        xlsx_path = data_root / "iara.xlsx"
        
        # Check if data exists
        h_folder = data_root / "H"
        if not h_folder.exists():
            print(f"⚠️  H folder not found at: {h_folder}")
            return False
        
        wav_files = list(h_folder.glob("*.wav"))
        print(f"✓ Found {len(wav_files)} WAV files in H folder")
        
        # Load metadata
        manager = MetadataManager(
            csv_path=csv_path if csv_path.exists() else None,
            xlsx_path=xlsx_path if xlsx_path.exists() else None,
            data_root=data_root
        )
        
        df = manager.load_metadata()
        print(f"✓ Loaded metadata: {len(df)} recordings")
        
        # Filter to H
        h_df = manager.filter_by_dc(["H"])
        print(f"✓ H folder has {len(h_df)} recordings in metadata")
        
        # Get statistics
        stats = manager.get_statistics()
        print(f"✓ Dataset statistics:")
        print(f"  - Total: {stats['total_recordings']}")
        print(f"  - Glider: {stats['glider_recordings']}")
        print(f"  - Background: {stats['background_recordings']}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test audio preprocessing."""
    print("\n" + "=" * 70)
    print("3. Testing Audio Preprocessing")
    print("=" * 70)
    
    try:
        from src.data.preprocessing import AudioPreprocessor, SpectrogramType
        import librosa
        
        data_root = project_root / "data" / "downloaded_content"
        h_folder = data_root / "H"
        
        if not h_folder.exists():
            print("⚠️  H folder not found")
            return False
        
        wav_files = list(h_folder.glob("*.wav"))
        if not wav_files:
            print("⚠️  No WAV files found")
            return False
        
        sample_file = str(wav_files[0])
        print(f"✓ Testing with: {Path(sample_file).name}")
        
        # Initialize preprocessor
        preprocessor = AudioPreprocessor(
            target_sr=16000,
            n_fft=1024,
            hop_length=1024,
            n_mels=128,
            averaging_windows=8,
        )
        
        # Load audio
        audio, sr = preprocessor.load_audio(sample_file)
        print(f"✓ Loaded audio: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.2f}s)")
        
        # Extract MEL
        mel_spec = preprocessor.extract_mel_spectrogram(audio)
        print(f"✓ MEL spectrogram: {mel_spec.shape}")
        
        # Extract LOFAR
        lofar_spec = preprocessor.extract_lofar_spectrogram(audio)
        print(f"✓ LOFAR spectrogram: {lofar_spec.shape}")
        
        # Create windows
        windows = preprocessor.create_image_windows(mel_spec, window_size=32, overlap=0.5)
        print(f"✓ Created {windows.shape[0]} windows (shape: {windows.shape})")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset creation."""
    print("\n" + "=" * 70)
    print("4. Testing Dataset Creation")
    print("=" * 70)
    
    try:
        from src.data.metadata import MetadataManager
        from src.data.preprocessing import AudioPreprocessor, SpectrogramType
        from src.data.dataset import IARAAudioDataset
        
        data_root = project_root / "data" / "downloaded_content"
        csv_path = project_root / "IARA" / "src" / "iara" / "dataset_info" / "iara.csv"
        xlsx_path = data_root / "iara.xlsx"
        
        # Initialize
        manager = MetadataManager(
            csv_path=csv_path if csv_path.exists() else None,
            xlsx_path=xlsx_path if xlsx_path.exists() else None,
            data_root=data_root
        )
        df = manager.load_metadata()
        
        preprocessor = AudioPreprocessor()
        
        # Create dataset
        dataset = IARAAudioDataset(
            data_root=data_root,
            metadata_manager=manager,
            preprocessor=preprocessor,
            feature_type=SpectrogramType.MEL,
            use_windows=True,
            dcs=["H"],
        )
        
        print(f"✓ Dataset created: {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Get first sample
            sample = dataset[0]
            print(f"✓ Sample 0: {sample.filename}")
            print(f"  - Features: {sample.features.shape}")
            print(f"  - Label: {sample.label} ({sample.class_name})")
            
            # Class weights
            weights = dataset.get_class_weights()
            print(f"✓ Class weights: {weights.numpy()}")
        else:
            print("⚠️  Dataset is empty")
        
        return True
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test model instantiation."""
    print("\n" + "=" * 70)
    print("5. Testing Model Instantiation")
    print("=" * 70)
    
    try:
        import torch
        from src.models.cnn import get_iara_cnn
        from src.models.mlp import get_iara_mlp
        from src.models.autoencoder import ConvolutionalAutoencoder
        
        # CNN
        cnn = get_iara_cnn(num_classes=4, config="mel")
        x = torch.randn(2, 1, 128, 32)
        out = cnn(x)
        print(f"✓ CNN: {x.shape} → {out.shape}")
        
        # MLP
        mlp = get_iara_mlp(input_size=128*100, num_classes=4, config="mel")
        x = torch.randn(2, 128*100)
        out = mlp(x)
        print(f"✓ MLP: {x.shape} → {out.shape}")
        
        # Autoencoder
        ae = ConvolutionalAutoencoder(latent_dim=64)
        x = torch.randn(2, 1, 128, 128)
        out = ae(x)
        print(f"✓ Autoencoder: {x.shape} → {out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "=" * 70)
    print("6. Testing Evaluation Metrics")
    print("=" * 70)
    
    try:
        import numpy as np
        from src.evaluation import compute_metrics, sum_product_index, MetricsTracker
        
        # Create dummy predictions
        np.random.seed(42)
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 1, 1, 2, 0, 0, 1])
        
        # Compute SPI
        sp = sum_product_index(y_true, y_pred, num_classes=4)
        print(f"✓ Sum-Product Index: {sp:.4f}")
        
        # Compute all metrics
        class_names = ["Small", "Medium", "Large", "Background"]
        metrics = compute_metrics(y_true, y_pred, 4, class_names)
        
        print(f"✓ Metrics computed:")
        print(f"  - SP: {metrics['sp']:.4f}")
        print(f"  - Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  - F1-Score (macro): {metrics['f1_macro']:.4f}")
        
        # Test tracker
        tracker = MetricsTracker(4, class_names)
        tracker.add_fold(y_true, y_pred)
        tracker.add_fold(y_pred, y_true)  # Swap for variation
        
        summary = tracker.get_summary()
        print(f"✓ MetricsTracker works (2 folds tracked)")
        
        return True
    except Exception as e:
        print(f"❌ Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("IARA ML Pipeline - Quick Verification")
    print("=" * 70)
    print("\nThis script tests all pipeline components with the H folder data.")
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Preprocessing", test_preprocessing),
        ("Dataset Creation", test_dataset),
        ("Model Instantiation", test_models),
        ("Evaluation Metrics", test_metrics),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status:10s} {test_name}")
    
    print("\n" + "-" * 70)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n✅ All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Run EDA: python3 notebooks/run_eda.py")
        print("  2. Train model: python3 scripts/train_baseline_cnn.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - H folder not found: Check data/downloaded_content/H/")
        print("  - Metadata not found: Check IARA/src/iara/dataset_info/iara.csv")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
