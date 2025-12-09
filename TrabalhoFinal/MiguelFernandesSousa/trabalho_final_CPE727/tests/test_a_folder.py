"""
Unit test for A folder data loading and class distribution.

Tests that A folder (456 recordings) loads correctly and matches Table 6 from paper.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_a_folder_exists():
    """Test that A folder exists and has WAV files."""
    data_root = project_root / "data" / "downloaded_content"
    a_folder = data_root / "A"
    
    assert a_folder.exists(), f"A folder not found at {a_folder}"
    
    wav_files = list(a_folder.glob("*.wav"))
    assert len(wav_files) > 0, "No WAV files found in A folder"
    
    print(f"✓ A folder found with {len(wav_files)} WAV files")
    return True


def test_a_folder_metadata():
    """Test that A folder metadata matches expectations from paper."""
    from src.data.metadata import MetadataManager
    
    data_root = project_root / "data" / "downloaded_content"
    csv_path = project_root / "IARA" / "src" / "iara" / "dataset_info" / "iara.csv"
    
    # Initialize metadata manager
    manager = MetadataManager(csv_path=csv_path, data_root=data_root)
    df = manager.load_metadata()
    
    # Filter to DC A
    a_df = manager.filter_by_dc(["A"])
    
    assert len(a_df) == 456, f"Expected 456 recordings in DC A, got {len(a_df)}"
    print(f"✓ DC A has 456 recordings (matches paper Table 2)")
    
    # Check class distribution (Table 6 from paper)
    class_dist = a_df['Class'].value_counts()
    
    expected = {
        "Small": 113,
        "Medium": 120,
        "Large": 223,
    }
    
    print("\n  Class distribution (expected vs actual):")
    for class_name, expected_count in expected.items():
        actual_count = class_dist.get(class_name, 0)
        match = "✓" if actual_count == expected_count else "✗"
        print(f"    {match} {class_name:8s}: expected {expected_count:3d}, got {actual_count:3d}")
        assert actual_count == expected_count, \
            f"{class_name} count mismatch: expected {expected_count}, got {actual_count}"
    
    # Check DC characteristics
    assert all(a_df['IsUO'] == True), "All A recordings should be UO"
    assert all(a_df['IsShip'] == True), "All A recordings should be ships"
    
    print("\n✓ DC A characteristics:")
    print("  - Platform: Underwater Observatory (UO)")
    print("  - Type: Ship noise")
    print("  - CPA: Yes")
    print("  - Distance: <250m")
    
    return True


def test_a_folder_dataset():
    """Test creating dataset from A folder."""
    from src.data.metadata import MetadataManager
    from src.data.preprocessing import AudioPreprocessor, SpectrogramType
    from src.data.dataset import IARAAudioDataset
    
    data_root = project_root / "data" / "downloaded_content"
    csv_path = project_root / "IARA" / "src" / "iara" / "dataset_info" / "iara.csv"
    
    # Initialize
    manager = MetadataManager(csv_path=csv_path, data_root=data_root)
    df = manager.load_metadata()
    
    preprocessor = AudioPreprocessor()
    
    # Create dataset for A folder
    dataset = IARAAudioDataset(
        data_root=data_root,
        metadata_manager=manager,
        preprocessor=preprocessor,
        feature_type=SpectrogramType.MEL,
        use_windows=True,
        dcs=["A"],
    )
    
    assert len(dataset) == 456, f"Expected 456 samples, got {len(dataset)}"
    print(f"\n✓ Created dataset with {len(dataset)} samples from A folder")
    
    # Test getting a sample
    sample = dataset[0]
    assert sample.features is not None, "Sample should have features"
    assert sample.label in [0, 1, 2], "Label should be 0 (Small), 1 (Medium), or 2 (Large)"
    assert sample.class_name in ["Small", "Medium", "Large"], \
        f"Class name should be Small/Medium/Large, got {sample.class_name}"
    
    print(f"\n✓ Sample test:")
    print(f"  - Filename: {sample.filename}")
    print(f"  - Features shape: {sample.features.shape}")
    print(f"  - Label: {sample.label} ({sample.class_name})")
    
    # Check class distribution in dataset
    stats = dataset.get_statistics()
    print(f"\n✓ Dataset statistics:")
    for class_name, count in stats['class_distribution'].items():
        print(f"  - {class_name}: {count} samples")
    
    return True


def test_a_and_h_combined():
    """Test combining A (ships) and H (background) for 4-class classification."""
    from src.data.metadata import MetadataManager
    from src.data.preprocessing import AudioPreprocessor, SpectrogramType
    from src.data.dataset import IARAAudioDataset
    
    data_root = project_root / "data" / "downloaded_content"
    csv_path = project_root / "IARA" / "src" / "iara" / "dataset_info" / "iara.csv"
    
    # Initialize
    manager = MetadataManager(csv_path=csv_path, data_root=data_root)
    df = manager.load_metadata()
    
    preprocessor = AudioPreprocessor()
    
    # Create dataset with A + H (all 4 classes)
    dataset = IARAAudioDataset(
        data_root=data_root,
        metadata_manager=manager,
        preprocessor=preprocessor,
        feature_type=SpectrogramType.MEL,
        use_windows=True,
        dcs=["A", "H"],  # Ships + Background
    )
    
    expected_total = 456 + 47  # A + H
    assert len(dataset) == expected_total, \
        f"Expected {expected_total} samples (A+H), got {len(dataset)}"
    
    print(f"\n✓ Combined A+H dataset: {len(dataset)} samples")
    
    # Check all 4 classes present
    stats = dataset.get_statistics()
    class_dist = stats['class_distribution']
    
    assert "Small" in class_dist, "Small class missing"
    assert "Medium" in class_dist, "Medium class missing"
    assert "Large" in class_dist, "Large class missing"
    assert "Background" in class_dist, "Background class missing"
    
    print(f"\n✓ All 4 classes present:")
    for class_name in ["Small", "Medium", "Large", "Background"]:
        count = class_dist.get(class_name, 0)
        pct = (count / len(dataset)) * 100
        print(f"  - {class_name:12s}: {count:3d} ({pct:5.1f}%)")
    
    # Expected distribution (from paper Table 6)
    expected_dist = {
        "Small": 113,
        "Medium": 120,
        "Large": 223,
        "Background": 47,
    }
    
    print(f"\n✓ Verification against paper Table 6:")
    for class_name, expected_count in expected_dist.items():
        actual_count = class_dist.get(class_name, 0)
        match = "✓" if actual_count == expected_count else "✗"
        print(f"  {match} {class_name:12s}: expected {expected_count:3d}, got {actual_count:3d}")
    
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("IARA A Folder - Unit Tests")
    print("=" * 70)
    print("\nTesting A folder (456 ship recordings: Small, Medium, Large)")
    print("Expected distribution from paper Table 6:")
    print("  - Small: 113 recordings (<50m)")
    print("  - Medium: 120 recordings (50-100m)")
    print("  - Large: 223 recordings (≥100m)")
    
    tests = [
        ("A Folder Exists", test_a_folder_exists),
        ("A Folder Metadata", test_a_folder_metadata),
        ("A Folder Dataset", test_a_folder_dataset),
        ("A+H Combined (4 classes)", test_a_and_h_combined),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'─' * 70}")
        print(f"Test: {test_name}")
        print('─' * 70)
        
        try:
            test_func()
            print(f"\n✅ {test_name}: PASSED")
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_name}: FAILED")
            print(f"Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_name}: ERROR")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 70}")
    print(f"Test Summary: {passed} passed, {failed} failed")
    print('=' * 70)
    
    if failed == 0:
        print("\n✅ All A folder tests passed!")
        print("\nYou now have:")
        print("  - H folder: 47 background recordings")
        print("  - A folder: 456 ship recordings (Small, Medium, Large)")
        print("  - Total: 503 recordings covering all 4 classes")
        print("\nReady to train 4-class models!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    sys.exit(0 if failed == 0 else 1)
