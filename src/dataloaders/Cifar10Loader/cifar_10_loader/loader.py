import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split

# Estatísticas do CIFAR-10 (R, G, B)
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_tf = T.Compose([T.ToTensor()])

class LabelNoiseDataset(torch.utils.data.Dataset):
    """Aplica ruído de rótulo com prob p_noise."""
    def __init__(self, base_ds, p_noise: float, num_classes: int = 10, seed: int = 0):
        self.base = base_ds
        self.p = float(p_noise)
        self.C = int(num_classes)
        rng = np.random.RandomState(seed)
        self.flip_mask = rng.rand(len(self.base)) < self.p
        self.rand_labels = rng.randint(0, self.C, size=len(self.base))

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if self.flip_mask[i]:
            y = int(self.rand_labels[i])
        return x, y

class TestNoiseWrapper(torch.utils.data.Dataset):
    """
    Adiciona ruído gaussiano em pixel space, faz clamp [0,1] e depois
    normaliza com mean/std.
    """
    def __init__(self, base_ds, sigma: float, mean_=mean, std_=std):
        self.base, self.sigma = base_ds, float(sigma)
        self.normalize = T.Normalize(mean_, std_)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]  # tensor em [0,1]
        if self.sigma > 0:
            noise = torch.randn_like(x) * self.sigma
            x = torch.clamp(x + noise, 0.0, 1.0)
        x = self.normalize(x)
        return x, y

def _split_with_seed(dataset, val_size: int, seed: int = 0):
    N = len(dataset)
    if val_size >= N:
        raise ValueError("val_size >= dataset size")
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [N - val_size, val_size], generator=gen)

def build_datasets(
    data_root: str = "./data",
    subset_train: int | None = None,
    p_label_noise: float = 0.0,
    val_split: int = 5000,
    seed: int = 0,
):
    # Base train/test
    train_raw = tv.datasets.CIFAR10(root=data_root, train=True,  download=True, transform=train_tf)
    test_set  = tv.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    # Subamostragem
    if subset_train is not None and subset_train < len(train_raw):
        idx = np.random.RandomState(seed).permutation(len(train_raw))[:subset_train]
        train_raw = Subset(train_raw, idx)

    # Ruído de rótulo
    noisy_train = LabelNoiseDataset(train_raw, p_noise=p_label_noise, num_classes=10, seed=seed)

    # Split train/val reprodutível
    train_part, val_part = _split_with_seed(noisy_train, val_split, seed=seed)
    return train_part, val_part, test_set

def build_loaders(
    train_part,
    val_part,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    train_loader = DataLoader(train_part, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_part,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

def make_noisy_test_loader(
    base_test_ds,
    sigma: float,
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    wrapped = TestNoiseWrapper(base_test_ds, sigma)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)
