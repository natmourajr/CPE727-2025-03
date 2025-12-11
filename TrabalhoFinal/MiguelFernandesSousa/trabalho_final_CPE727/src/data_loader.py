"""
Data loader para Fashion MNIST

O dataset Fashion MNIST é automaticamente baixado e armazenado em cache
pela biblioteca torchvision, com fallback para download manual via GitHub mirror.
"""
import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np

# PyTorch import - may fail on Python 3.13.1 due to compatibility issue
# If import fails, we'll handle it in the class
try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except (AttributeError, ImportError) as e:
    # Handle PyTorch compatibility issues
    TORCH_AVAILABLE = False
    print(f"Warning: PyTorch import failed ({e}).")
    print("This project requires Python 3.10-3.12 with PyTorch 2.0-2.4.")
    print("Python 3.13+ is not yet supported by PyTorch.")
    raise

from sklearn.model_selection import train_test_split

from src.config import FASHION_MNIST_DIR, BATCH_SIZE, RANDOM_SEED


def _download_fashion_mnist_manual(data_dir: Path):
    """Download Fashion MNIST from GitHub mirror if torchvision download fails"""
    base_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    raw_dir = data_dir / "FashionMNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Fashion MNIST from GitHub mirror...")
    for filename in files:
        filepath = raw_dir / filename
        if not filepath.exists():
            url = base_url + filename
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ {filename} downloaded")


class FashionMNISTLoader:
    """
    Classe para carregar e pré-processar o dataset Fashion MNIST
    """

    def __init__(self, flatten=True, normalize=True, normalization_range='-1_1', batch_size=BATCH_SIZE):
        """
        Args:
            flatten: Se True, transforma imagens 28x28 em vetores de 784 elementos
            normalize: Se True, normaliza pixels
            normalization_range: Range de normalização:
                '-1_1': Normaliza para [-1, +1] (padrão, conforme proposta)
                '0_1': Normaliza para [0, 1] (necessário para MultinomialNB)
            batch_size: Tamanho do batch para DataLoader
        """
        self.flatten = flatten
        self.normalize = normalize
        self.normalization_range = normalization_range
        self.batch_size = batch_size
        self.random_seed = RANDOM_SEED

        # Validar normalization_range
        if self.normalization_range not in ['-1_1', '0_1']:
            raise ValueError(f"normalization_range must be '-1_1' or '0_1', got '{self.normalization_range}'")

        # Transformações
        transform_list = []
        if normalize:
            transform_list.append(transforms.ToTensor())  # Converte para tensor e [0, 1]
        else:
            transform_list.append(transforms.PILToTensor())

        self.transform = transforms.Compose(transform_list)

        # Tentar carregar datasets
        try:
            print("Loading Fashion MNIST...")
            self.train_dataset = datasets.FashionMNIST(
                root=str(FASHION_MNIST_DIR),
                train=True,
                download=True,
                transform=self.transform,
            )
            self.test_dataset = datasets.FashionMNIST(
                root=str(FASHION_MNIST_DIR),
                train=False,
                download=True,
                transform=self.transform,
            )
        except (RuntimeError, urllib.error.HTTPError) as e:
            print(f"Standard download failed ({e}), using GitHub mirror...")
            _download_fashion_mnist_manual(FASHION_MNIST_DIR)

            # Tentar novamente COM download=True para processar os arquivos baixados
            self.train_dataset = datasets.FashionMNIST(
                root=str(FASHION_MNIST_DIR),
                train=True,
                download=True,
                transform=self.transform,
            )
            self.test_dataset = datasets.FashionMNIST(
                root=str(FASHION_MNIST_DIR),
                train=False,
                download=True,
                transform=self.transform,
            )

    def get_dataloaders(self, val_split=0.2):
        """
        Retorna DataLoaders para treino, validação e teste

        Args:
            val_split: Proporção do conjunto de treino para validação

        Returns:
            train_loader, val_loader, test_loader
        """
        # Criar validação a partir do treino
        train_size = len(self.train_dataset)
        indices = list(range(train_size))
        train_indices, val_indices = train_test_split(
            indices, test_size=val_split, random_state=self.random_seed
        )

        train_subset = torch.utils.data.Subset(self.train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(self.train_dataset, val_indices)

        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        return train_loader, val_loader, test_loader

    def get_numpy_arrays(self, val_split=0.2):
        """
        Retorna dados como arrays numpy (útil para scikit-learn)

        Args:
            val_split: Proporção do conjunto de treino para validação

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        # Converter para numpy
        X_train_full = self.train_dataset.data.numpy()
        y_train_full = self.train_dataset.targets.numpy()
        X_test = self.test_dataset.data.numpy()
        y_test = self.test_dataset.targets.numpy()

        # Normalizar se necessário
        if self.normalize:
            if self.normalization_range == '0_1':
                # Normalização [0, 1] (para MultinomialNB)
                X_train_full = X_train_full.astype(np.float32) / 255.0
                X_test = X_test.astype(np.float32) / 255.0
            elif self.normalization_range == '-1_1':
                # Normalização [-1, +1] (padrão conforme proposta)
                X_train_full = (X_train_full.astype(np.float32) / 255.0) * 2.0 - 1.0
                X_test = (X_test.astype(np.float32) / 255.0) * 2.0 - 1.0

        # Flatten se necessário
        if self.flatten:
            X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        # Split treino/validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_split, random_state=self.random_seed, stratify=y_train_full
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_class_names(self):
        """Retorna os nomes das classes"""
        from src.config import CLASS_NAMES

        return CLASS_NAMES

    def get_dataset_info(self):
        """Retorna informações sobre o dataset"""
        train_size = len(self.train_dataset)
        test_size = len(self.test_dataset)

        info = {
            "train_samples": train_size,
            "test_samples": test_size,
            "num_classes": 10,
            "image_shape": (28, 28),
            "feature_dim": 784 if self.flatten else (28, 28),
        }

        return info


if __name__ == "__main__":
    # Teste do data loader
    print("Carregando Fashion MNIST...")
    loader = FashionMNISTLoader(flatten=True, normalize=True)

    # Informações do dataset
    info = loader.get_dataset_info()
    print("\nInformações do dataset:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Testar carregamento como numpy
    print("\nCarregando dados como numpy arrays...")
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    print(f"\nShapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")

    # Testar DataLoaders
    print("\nCriando DataLoaders...")
    train_loader, val_loader, test_loader = loader.get_dataloaders()

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Pegar um batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch example:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")

    print("\n✓ Data loader funcionando corretamente!")
