import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2, Compose

class NoisyMNISTDataset(Dataset):
    """Dataset class for Noisy MNIST dataset"""

    def __init__(self, download_path, download=True, train=True, noise_level=0.1, noise_seed=42):
        """
        Args:
            download_path: Path to download/store the MNIST dataset
            download: Whether to download the dataset (default: True)
            train: Whether to load the training set (default: True)
        """
        self.noise_level = noise_level
        self.mnist_data = datasets.MNIST(
            root=download_path,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )
        self.X = self.y = self.mnist_data.data.float() / 255.0  # Normalize to [0, 1]
        self.X = self.__add_noise(self.X, seed=noise_seed)
        self.y_classes = self.mnist_data.targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_classes[idx]

    def __add_noise(self, X, seed=42):
        """Private method to add noise to the images"""
        torch.manual_seed(seed)
        noise_transform = Compose([
            v2.Lambda(lambda img: img + self.noise_level * torch.randn_like(img)),
            v2.Lambda(lambda img: torch.clamp(img, 0.0, 1.0))  # Ensure pixel values are in [0, 1]
        ])
        return noise_transform(X)
