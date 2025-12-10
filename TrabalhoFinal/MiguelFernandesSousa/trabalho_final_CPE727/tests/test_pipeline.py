"""
Unit tests for IARA ML pipeline.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

# Test data loading
def test_metadata_loading():
    """Test metadata manager can load and process metadata."""
    from src.data.metadata import MetadataManager
    
    # Use actual paths
    data_root = Path("data/downloaded_content")
    csv_path = Path("IARA/src/iara/dataset_info/iara.csv")
    xlsx_path = Path("data/downloaded_content/iara.xlsx")
    
    # Try CSV first, then XLSX
    if csv_path.exists():
        metadata_path = csv_path
    elif xlsx_path.exists():
        metadata_path = xlsx_path
    else:
        pytest.skip("No metadata file found")
    
    manager = MetadataManager(
        csv_path=csv_path if csv_path.exists() else None,
        xlsx_path=xlsx_path if xlsx_path.exists() else None,
        data_root=data_root
    )
    
    df = manager.load_metadata()
    
    assert len(df) > 0, "Metadata should not be empty"
    assert "Class" in df.columns, "Should have Class column"
    assert "ClassIdx" in df.columns, "Should have ClassIdx column"
    
    # Check class distribution
    stats = manager.get_statistics()
    assert "total_recordings" in stats
    assert stats["total_recordings"] > 0
    
    print(f"✓ Loaded {len(df)} recordings")
    print(f"✓ Class distribution: {stats['class_distribution']}")


def test_audio_preprocessing():
    """Test audio preprocessing with actual H folder sample."""
    from src.data.preprocessing import AudioPreprocessor, SpectrogramType
    
    # Find a sample WAV file from H folder
    data_root = Path("data/downloaded_content/H")
    
    if not data_root.exists():
        pytest.skip("H folder not found")
    
    wav_files = list(data_root.glob("*.wav"))
    if not wav_files:
        pytest.skip("No WAV files found in H folder")
    
    sample_file = str(wav_files[0])
    print(f"\nTesting with: {Path(sample_file).name}")
    
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
    assert len(audio) > 0, "Audio should not be empty"
    assert sr == 16000, "Sample rate should be 16kHz"
    
    print(f"✓ Loaded audio: {len(audio)} samples at {sr}Hz")
    
    # Extract MEL spectrogram
    mel_spec = preprocessor.extract_mel_spectrogram(audio)
    assert mel_spec.shape[0] == 128, "Should have 128 mel bands"
    assert mel_spec.shape[1] > 0, "Should have time steps"
    
    print(f"✓ MEL spectrogram shape: {mel_spec.shape}")
    
    # Extract LOFAR spectrogram
    lofar_spec = preprocessor.extract_lofar_spectrogram(audio)
    assert lofar_spec.shape[1] > 0, "Should have time steps"
    
    print(f"✓ LOFAR spectrogram shape: {lofar_spec.shape}")
    
    # Test window creation for CNN
    windows = preprocessor.create_image_windows(mel_spec, window_size=32, overlap=0.5)
    assert windows.shape[2] == 32, "Windows should have size 32"
    
    print(f"✓ Created {windows.shape[0]} windows for CNN")


def test_dataset_creation():
    """Test IARA dataset creation."""
    from src.data.metadata import MetadataManager
    from src.data.preprocessing import AudioPreprocessor, SpectrogramType
    from src.data.dataset import IARAAudioDataset
    
    # Setup paths
    data_root = Path("data/downloaded_content")
    csv_path = Path("IARA/src/iara/dataset_info/iara.csv")
    xlsx_path = Path("data/downloaded_content/iara.xlsx")
    
    if not data_root.exists():
        pytest.skip("Data folder not found")
    
    # Initialize metadata manager
    manager = MetadataManager(
        csv_path=csv_path if csv_path.exists() else None,
        xlsx_path=xlsx_path if xlsx_path.exists() else None,
        data_root=data_root
    )
    df = manager.load_metadata()
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Create dataset for H DC only (background noise)
    dataset = IARAAudioDataset(
        data_root=data_root,
        metadata_manager=manager,
        preprocessor=preprocessor,
        feature_type=SpectrogramType.MEL,
        use_windows=False,
        dcs=["H"],  # Only H DC
    )
    
    assert len(dataset) > 0, "Dataset should have samples"
    
    print(f"\n✓ Created dataset with {len(dataset)} samples from H DC")
    
    # Test getting a sample
    sample = dataset[0]
    assert sample.features is not None, "Sample should have features"
    assert sample.label is not None, "Sample should have label"
    assert sample.class_name is not None, "Sample should have class name"
    
    print(f"✓ Sample: {sample.filename}")
    print(f"  - Features shape: {sample.features.shape}")
    print(f"  - Label: {sample.label} ({sample.class_name})")
    
    # Test class weights
    weights = dataset.get_class_weights()
    assert len(weights) > 0, "Should have class weights"
    
    print(f"✓ Class weights computed: {weights.numpy()}")


def test_evaluation_metrics():
    """Test evaluation metrics computation."""
    from src.evaluation import (
        sum_product_index,
        compute_metrics,
        format_metrics,
        MetricsTracker,
    )
    
    # Create dummy predictions
    num_samples = 100
    num_classes = 4
    
    y_true = np.random.randint(0, num_classes, size=num_samples)
    y_pred = np.random.randint(0, num_classes, size=num_samples)
    
    # Compute SP
    sp = sum_product_index(y_true, y_pred, num_classes)
    assert 0 <= sp <= 1, "SP should be between 0 and 1"
    
    print(f"\n✓ Sum-Product Index: {sp:.4f}")
    
    # Compute all metrics
    class_names = ["Small", "Medium", "Large", "Background"]
    metrics = compute_metrics(y_true, y_pred, num_classes, class_names)
    
    assert "sp" in metrics
    assert "balanced_accuracy" in metrics
    assert "f1_macro" in metrics
    
    print("✓ Metrics computed:")
    print(format_metrics(metrics))
    
    # Test metrics tracker
    tracker = MetricsTracker(num_classes, class_names)
    
    # Add multiple folds
    for fold in range(5):
        y_true_fold = np.random.randint(0, num_classes, size=num_samples)
        y_pred_fold = np.random.randint(0, num_classes, size=num_samples)
        tracker.add_fold(y_true_fold, y_pred_fold)
    
    summary = tracker.get_summary()
    assert "sp" in summary
    assert "balanced_accuracy" in summary
    
    print("\n✓ Metrics tracker summary (5 folds):")
    print(tracker.format_summary())


def test_model_imports():
    """Test that all models can be imported and instantiated."""
    from src.models import (
        IARACNN,
        IARAMLP,
        get_resnet18,
        get_efficientnet_b0,
        get_convnext_tiny,
        ConvolutionalAutoencoder,
        DenoisingAutoencoder,
    )
    
    num_classes = 4
    batch_size = 2
    
    print("\n✓ Testing model instantiation:")
    
    # Test CNN
    cnn = IARACNN(num_classes=num_classes)
    x = torch.randn(batch_size, 1, 128, 32)
    out = cnn(x)
    assert out.shape == (batch_size, num_classes)
    print(f"  - CNN: input {x.shape} → output {out.shape}")
    
    # Test MLP
    input_size = 128 * 100
    mlp = IARAMLP(input_size=input_size, num_classes=num_classes)
    x = torch.randn(batch_size, input_size)
    out = mlp(x)
    assert out.shape == (batch_size, num_classes)
    print(f"  - MLP: input {x.shape} → output {out.shape}")
    
    # Test ResNet (if timm available)
    try:
        resnet = get_resnet18(num_classes=num_classes, pretrained=False)
        x = torch.randn(batch_size, 1, 128, 128)
        out = resnet(x)
        assert out.shape == (batch_size, num_classes)
        print(f"  - ResNet18: input {x.shape} → output {out.shape}")
    except ImportError:
        print("  - ResNet18: skipped (timm not installed)")
    
    # Test EfficientNet (if timm available)
    try:
        efficientnet = get_efficientnet_b0(num_classes=num_classes, pretrained=False)
        x = torch.randn(batch_size, 1, 128, 128)
        out = efficientnet(x)
        assert out.shape == (batch_size, num_classes)
        print(f"  - EfficientNet-B0: input {x.shape} → output {out.shape}")
    except ImportError:
        print("  - EfficientNet-B0: skipped (timm not installed)")
    
    # Test ConvNeXt (if timm available)
    try:
        convnext = get_convnext_tiny(num_classes=num_classes, pretrained=False)
        x = torch.randn(batch_size, 1, 128, 128)
        out = convnext(x)
        assert out.shape == (batch_size, num_classes)
        print(f"  - ConvNeXt-Tiny: input {x.shape} → output {out.shape}")
    except ImportError:
        print("  - ConvNeXt-Tiny: skipped (timm not installed)")
    
    # Test Autoencoder
    autoencoder = ConvolutionalAutoencoder(latent_dim=64)
    x = torch.randn(batch_size, 1, 128, 128)
    out = autoencoder(x)
    print(f"  - Autoencoder: input {x.shape} → output {out.shape}")
    
    # Test Denoising Autoencoder
    denoising_ae = DenoisingAutoencoder(latent_dim=64, noise_factor=0.3)
    x = torch.randn(batch_size, 1, 128, 128)
    x_noisy, out = denoising_ae(x)
    print(f"  - Denoising AE: input {x.shape} → output {out.shape}")
    
    print("✓ All models instantiated successfully!")


if __name__ == "__main__":
    print("=" * 70)
    print("IARA ML Pipeline Unit Tests")
    print("=" * 70)
    
    tests = [
        ("Metadata Loading", test_metadata_loading),
        ("Audio Preprocessing", test_audio_preprocessing),
        ("Dataset Creation", test_dataset_creation),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Model Imports", test_model_imports),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        print(f"\n{'─' * 70}")
        print(f"Test: {test_name}")
        print('─' * 70)
        
        try:
            test_func()
            print(f"\n✅ {test_name}: PASSED")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"\n⚠️  {test_name}: SKIPPED - {e}")
            skipped += 1
        except Exception as e:
            print(f"\n❌ {test_name}: FAILED")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 70}")
    print(f"Test Summary: {passed} passed, {failed} failed, {skipped} skipped")
    print('=' * 70)
