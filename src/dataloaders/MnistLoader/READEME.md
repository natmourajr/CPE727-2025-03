# MNIST Loader

Pacote para carregar CIFAR-10 com:
- **Ruído de rótulo** no treino (`LabelNoiseDataset`)
- **Ruído de entrada** no teste **antes da normalização** (`TestNoiseWrapper`)
- Funções de fábrica para **datasets** e **DataLoaders**

## Instalação
```bash
pip install -e ./MnistLoader
```

# Uso rápido

```python
from mnist_loader.loader import build_loaders as MnistLoader

# Dataloaders
train_loader, test_loader = MnistLoader(
            root=save_path, # path to download the dataset
            transforms=transforms, # transformations to be applied on the dataset to DataAugmentation and Preprocess
            batch_size=batch_size, # size of the batches
            dataset_fraction=dataset_fraction # choose the fraction of the dataset to use in training, if None use full training dataset
        )

