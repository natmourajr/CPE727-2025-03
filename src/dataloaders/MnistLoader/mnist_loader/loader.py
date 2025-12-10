from torch.utils.data import Dataset, random_split, DataLoader
from torch import Generator
from torchvision import datasets, transforms

class MnistDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, download: bool = True):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CustomMnistDataset(Dataset):
    def __init__(self, root: str, train_split: float | None, train: bool = True, transform=None, download: bool = True, dataset_fraction: float | None = None):
        train_dataset = MnistDataset(
            root=root,
            train=True,
            transform=transform,
            download=download
        )
        test_dataset = MnistDataset(
            root=root,
            train=False,
            transform=transform,
            download=download
        )
        if train_split is not None:
            total_size = len(train_dataset)
            train_size = int(total_size * train_split)
            test_size = total_size - train_size
            train_dataset, test_dataset = random_split(
                train_dataset,
                [train_size, test_size],
                generator=Generator().manual_seed(42)
            )

        # Reduzir o tamanho do dataset, se solicitado
        if dataset_fraction is not None and 0 < dataset_fraction < 1:
            target_dataset = train_dataset if train else test_dataset
            reduced_size = int(len(target_dataset) * dataset_fraction)
            target_dataset, _ = random_split(
                target_dataset,
                [reduced_size, len(target_dataset) - reduced_size],
                generator=Generator().manual_seed(42)
            )
            self.dataset = target_dataset
        else:
            self.dataset = train_dataset if train else test_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def build_loaders(
        root: str,
        transforms: transforms.Compose,
        batch_size: int,
        train_split: float | None = None,
        dataset_fraction: float | None = None
):
    train_dataset = CustomMnistDataset(
        root=root,
        train_split=train_split,
        train=True,
        transform=transforms,
        download=True,
        dataset_fraction=dataset_fraction
    )
    test_dataset = CustomMnistDataset(
        root=root,
        train_split=train_split,
        train=False,
        transform=transforms,
        download=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
