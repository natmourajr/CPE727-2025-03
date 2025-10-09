from torch.utils.data import Dataset
import torch
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


class WineQualityDataset(Dataset):
    """Dataset class for UCI Wine Quality dataset"""

    def __init__(self, split="train", train_ratio=0.8, random_state=42):
        """
        Args:
            split: 'train' or 'test'
            train_ratio: Proportion for train/test split (default: 0.8)
            random_state: Random seed for reproducibility (default: 42)
        """
        # Fetch from UCI ML Repository (cached by ucimlrepo internally)
        wine_quality = fetch_ucirepo(id=186)
        X = wine_quality.data.features
        y = wine_quality.data.targets

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_ratio, random_state=random_state
        )

        if split == "train":
            self.X = torch.FloatTensor(X_train.values)
            self.y = torch.FloatTensor(y_train.values)
        else:
            self.X = torch.FloatTensor(X_test.values)
            self.y = torch.FloatTensor(y_test.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
