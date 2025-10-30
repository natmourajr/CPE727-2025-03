# CIFAR-10 Loader

Pacote para carregar CIFAR-10 com:
- **Ruído de rótulo** no treino (`LabelNoiseDataset`)
- **Ruído de entrada** no teste **antes da normalização** (`TestNoiseWrapper`)
- Funções de fábrica para **datasets** e **DataLoaders**

## Instalação
```bash
pip install -e ./Cifar10Loader
```

# Uso rápido

```python
import torch
from cifar10_loader import (
    mean, std,
    build_datasets, build_loaders, make_noisy_test_loader
)

SEED = 0
BATCH = 128
NUM_WORKERS = 2

# Datasets (subset opcional + ruído de rótulo no treino)
train_part, val_part, test_set = build_datasets(
    data_root="./data",
    subset_train=20000,        # ou None
    p_label_noise=0.2,         # prob. de trocar o rótulo por classe aleatória
    val_split=5000,
    seed=SEED,
)

# Loaders de treino/validação
train_loader, val_loader = build_loaders(
    train_part, val_part,
    batch_size=BATCH,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Loader de teste com ruído de entrada
test_loader_sigma_01 = make_noisy_test_loader(
    test_set, sigma=0.1, batch_size=256, num_workers=NUM_WORKERS
)

# Iteração típica
for x, y in train_loader:
    # x já está normalizado (mean/std do CIFAR-10)
    ...

```