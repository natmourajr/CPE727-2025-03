from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


class BreastCancerDataset(Dataset):
    """
    Dataset class for UCI Breast Cancer dataset.

    Returns raw pandas DataFrames (not tensors) since features are categorical.
    Use breast_cancer_preprocessing module to encode features before training.
    """

    def __init__(self, split="train", train_ratio=0.8, random_state=42):
        """
        Args:
            split: 'train' or 'test'
            train_ratio: Proportion for train/test split (default: 0.8)
            random_state: Random seed for reproducibility (default: 42)
        """
        # Fetch from UCI ML Repository (cached by ucimlrepo internally)
        breast_cancer = fetch_ucirepo(id=14)
        X = breast_cancer.data.features
        y = breast_cancer.data.targets

        # Split data - keep as pandas DataFrames
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_ratio, random_state=random_state
        )

        if split == "train":
            self.X = X_train.reset_index(drop=True)
            self.y = y_train.reset_index(drop=True)
        else:
            self.X = X_test.reset_index(drop=True)
            self.y = y_test.reset_index(drop=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Returns raw pandas Series (not tensors)"""
        return self.X.iloc[idx], self.y.iloc[idx]