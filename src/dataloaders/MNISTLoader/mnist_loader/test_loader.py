from mnist_loader.loader import MNISTDataset
import os

def test_mnist_dataset_train():
    """Test MNIST dataset loading for training set."""
    dataset = MNISTDataset(download_path="../../../Data", download=True, train=True)
    assert len(dataset) > 0
    X, y = dataset[0]
    assert X.shape == (28, 28)  # MNIST images are 28x28
    assert y.shape == ()        # y is a scalar label

def test_mnist_dataset_test():
    """Test MNIST dataset loading for test set."""
    dataset = MNISTDataset(download_path="../../../Data", download=True, train=False)
    assert len(dataset) > 0
    X, y = dataset[0]
    assert X.shape == (28, 28)  # MNIST images are 28x28
    assert y.shape == ()        # y is a scalar label

def test_mnist_dataset_max_min_values():
    """Test MNIST dataset for max and min pixel values."""
    dataset = MNISTDataset(download_path="../../../Data", download=True, train=False)
    X, y = dataset[0]
    assert X.max() <= 1  # Max pixel value should be 1
    assert X.min() >= 0    # Min pixel value should be 0

def test_mnist_dataset_classes():
    """Test MNIST dataset for correct number of classes."""
    dataset = MNISTDataset(download_path="../../../Data", download=True, train=False)
    mnist_targets = dataset.y
    assert set(mnist_targets.numpy()) == set(range(10))  # MNIST has 10 classes (0-9)
