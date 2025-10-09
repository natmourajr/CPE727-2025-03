# Pytest file for testing the WineQualityLoader

from wine_quality_loader.loader import WineQualityDataset


def test_wine_quality_dataset_train():
    dataset = WineQualityDataset(split="train", train_ratio=0.8, random_state=42)
    assert len(dataset) > 0
    X, y = dataset[0]
    assert X.shape[0] == 11  # 11 features in wine quality dataset
    assert y.shape[0] == 1   # 1 target (quality score)


def test_wine_quality_dataset_test():
    dataset = WineQualityDataset(split="test", train_ratio=0.8, random_state=42)
    assert len(dataset) > 0
    X, y = dataset[0]
    assert X.shape[0] == 11
    assert y.shape[0] == 1


def test_train_test_split_consistency():
    """Test that train and test splits are complementary and consistent"""
    train_dataset = WineQualityDataset(split="train", train_ratio=0.8, random_state=42)
    test_dataset = WineQualityDataset(split="test", train_ratio=0.8, random_state=42)

    # Check that splits are non-overlapping sizes
    total_samples = len(train_dataset) + len(test_dataset)
    assert total_samples > 0
    assert len(train_dataset) > len(test_dataset)  # 80/20 split


def test_reproducibility():
    """Test that same random_state produces same split"""
    dataset1 = WineQualityDataset(split="train", random_state=42)
    dataset2 = WineQualityDataset(split="train", random_state=42)

    assert len(dataset1) == len(dataset2)
    # Check first sample is identical
    X1, y1 = dataset1[0]
    X2, y2 = dataset2[0]
    assert (X1 == X2).all()
    assert (y1 == y2).all()
