# MNIST Loader

A PyTorch DataLoader for the MNIST dataset with support for both clean and noisy versions of the data.

## Description

This Data Loader provides a simple interface to load the MNIST (Modified National Institute of Standards and Technology) dataset, which contains grayscale images of handwritten digits (0-9) at 28x28 pixels.

The loader offers two main classes:

1. **MNISTDataset**: Loads the standard MNIST dataset
2. **NoisyMNISTDataset**: Loads the MNIST dataset with added Gaussian noise

## Features

- Automatic normalization of pixel values to the [0, 1] range
- Support for training and test sets
- Automatic dataset download when needed
- Option to add controlled Gaussian noise to images
- Unit tests included

## Usage

### MNISTDataset (Standard Dataset)

```python
from mnist_loader import MNISTDataset

# Load training set
train_dataset = MNISTDataset(
    download_path="./Data",
    download=True,  # Will only download if data is not already in download_path
    train=True
)

# Load test set
test_dataset = MNISTDataset(
    download_path="./Data",
    download=True,  # Will only download if data is not already in download_path
    train=False
)

# Access a sample
image, label = train_dataset[0]
print(f"Image shape: {image.shape}")  # (28, 28)
print(f"Label: {label}")  # value between 0 and 9
```

### NoisyMNISTDataset (Dataset with Noise)

```python
from mnist_loader.noise_mnist import NoisyMNISTDataset

# Load training set with noise
noisy_dataset = NoisyMNISTDataset(
    download_path="./Data/NoisyMnist",
    download=True,  # Will only download if data is not already in download_path
    train=True,
    noise_level=0.1,  # Standard deviation (σ) of Gaussian noise
    noise_seed=42     # Seed for reproducibility
)

# Access a sample
noisy_image, clean_image = noisy_dataset[0]
print(f"Noisy image shape: {noisy_image.shape}")  # (28, 28)
print(f"Clean image shape: {clean_image.shape}")  # (28, 28)
```

**Noise Generation Process**:
The noise is generated using the following formula: `noisy_image = original_image + noise_level * N(0, 1)`

Where:
- `N(0, 1)` is a random sample from a standard Gaussian distribution (mean μ = 0, standard deviation σ = 1)
- `noise_level` scales the noise, effectively setting the standard deviation of the added noise
- After adding noise, pixel values are clamped to the [0, 1] range to ensure validity

This means the final noise has a Gaussian distribution with mean = 0 and standard deviation = `noise_level`.

## Parameters

### MNISTDataset

- `download_path` (str): Path to download/store the MNIST dataset
- `download` (bool, default=True): Whether to download the dataset. **Note**: The dataset will only be downloaded if it's not already available in the `download_path`
- `train` (bool, default=True): Whether to load the training set (True) or test set (False)

### NoisyMNISTDataset

- `download_path` (str): Path to download/store the MNIST dataset
- `download` (bool, default=True): Whether to download the dataset. **Note**: The dataset will only be downloaded if it's not already available in the `download_path`
- `train` (bool, default=True): Whether to load the training set (True) or test set (False)
- `noise_level` (float, default=0.1): Standard deviation (σ) of the Gaussian noise to be added. The noise is sampled from N(0, 1) and multiplied by this value
- `noise_seed` (int, default=42): Seed for reproducible noise generation

## Dataset Structure

- **Training**: 60,000 images
- **Test**: 10,000 images
- **Image size**: 28x28 pixels
- **Number of classes**: 10 (digits from 0 to 9)
- **Format**: Grayscale

## Running Tests

If you are using uv, simply execute as follows:

```bash
uv run --extra dev pytest
```

If you are using only basic `pip` to manage packages, create a virtualenv and execute the following steps:

```bash
pip install -e '.[dev]'
pytest
```


## Dependencies

- `torch>=2.8.0`
- `torchvision>=0.23.0`
- `matplotlib>=3.10.7`

For development and testing:
- `pytest>=8.4.1`

## References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [PyTorch MNIST Documentation](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
