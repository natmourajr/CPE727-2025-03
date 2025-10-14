from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    """Dataset class for MNIST dataset"""

    def __init__(self, download_path, download=True, train=True):
        """
        Args:
            download_path: Path to download/store the MNIST dataset
            download: Whether to download the dataset (default: True)
            train: Whether to load the training set (default: True)
        """
        self.mnist_data = datasets.MNIST(
            root=download_path,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
        self.X = self.mnist_data.data.float() / 255.0  # Normalize to [0, 1]
        self.y = self.mnist_data.targets.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]