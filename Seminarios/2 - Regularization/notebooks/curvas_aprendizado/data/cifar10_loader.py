"""
CIFAR-10 Data Loading and Preprocessing

Reference: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
"""

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


class CIFAR10DataModule:
    """
    CIFAR-10 data loading and preprocessing
    
    Args:
        data_dir (str): Directory to store/load CIFAR-10 dataset
        batch_size (int): Batch size for dataloaders
        val_split (float): Fraction of training data to use for validation
        num_workers (int): Number of workers for data loading
        data_cap_rate (int): Rate to cap dataset size (e.g., 20 means use 1/20 of data)
    """
    
    def __init__(self, data_dir='./data', batch_size=64, val_split=0.1, num_workers=2, data_cap_rate=1):
        self.name = "CIFAR-10"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.data_cap_rate = data_cap_rate
        
        # CIFAR-10 normalization values (computed on training set)
        # Reference: https://docs.pytorch.org/vision/main/transforms.html
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        
        self.transform = self._get_transforms()
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _get_transforms(self):
        """
        Define preprocessing transforms
        
        - Convert PIL Image to Tensor
        - Normalize using CIFAR-10 statistics
        - NO data augmentation (as per requirements)
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def prepare_data(self):
        """Download CIFAR-10 dataset if not already present"""
        print("Preparing CIFAR-10 dataset...")
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)
        print("Dataset ready!")
    
    def setup(self):
        """Setup train, validation, and test datasets"""
        print(f"Setting up datasets (validation split: {self.val_split}, data_cap_rate: {self.data_cap_rate})...")
        
        # Load full training set
        full_train = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=False
        )
        
        # Split into train and validation
        val_size = int(len(full_train) * self.val_split)
        train_size = len(full_train) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducibility
        )
        
        # Load test set
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=False
        )
        
        # Apply data capping if specified
        if self.data_cap_rate > 1:
            # Cap training set
            train_subset_size = int(len(self.train_dataset) / self.data_cap_rate)
            train_indices = torch.randperm(len(self.train_dataset), generator=torch.Generator().manual_seed(42))[:train_subset_size]
            self.train_dataset = Subset(self.train_dataset, train_indices.tolist())
            
            # Cap validation set
            val_subset_size = int(len(self.val_dataset) / self.data_cap_rate)
            val_indices = torch.randperm(len(self.val_dataset), generator=torch.Generator().manual_seed(42))[:val_subset_size]
            self.val_dataset = Subset(self.val_dataset, val_indices.tolist())
            
            # Cap test set
            test_subset_size = int(len(self.test_dataset) / self.data_cap_rate)
            test_indices = torch.randperm(len(self.test_dataset), generator=torch.Generator().manual_seed(42))[:test_subset_size]
            self.test_dataset = Subset(self.test_dataset, test_indices.tolist())
            
            print(f"Applied data cap rate: 1/{self.data_cap_rate}")
        
        print(f"Train size: {len(self.train_dataset):,}, Val size: {len(self.val_dataset):,}, Test size: {len(self.test_dataset):,}")
    
    def train_dataloader(self):
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False  # Set to False for CPU training
        )
    
    def val_dataloader(self):
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
    
    def test_dataloader(self):
        """Get test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

