import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold


class NEUDataset(Dataset):
    """
        root_dir/
        ├── Crazing/
        ├── Inclusion/
        ├── Patches/
        ├── Pitted_surface/
        ├── Rolled_in_scale/
        └── Scratches/
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Detecta as classes automaticamente (ordem alfabética)
        classes = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))]
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls_name in classes:
            class_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif")):
                    path = os.path.join(class_dir, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

        print(f"[NEUDataset] Total de amostras carregadas: {len(self.samples)}")
        print(f"[NEUDataset] Classes encontradas: {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # escala de cinza

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def default_transform():
    return transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ])



def get_dataloaders(root_dir, batch_size=8, train_ratio=0.8, seed=42):

    transform = default_transform()
    full_dataset = NEUDataset(root_dir=root_dir, transform=transform)

    n_total = len(full_dataset)
    n_train = int(train_ratio * n_total)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[get_dataloaders] Train: {len(train_ds)} | Val: {len(val_ds)}")

    return train_loader, val_loader


def get_kfold_loaders(root_dir, batch_size=8, n_splits=5, seed=42):
 
    transform = default_transform()
    dataset = NEUDataset(root_dir=root_dir, transform=transform)

    # Rótulos para estratificação
    labels = [label for _, label in dataset.samples]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        print(f"[get_kfold_loaders] Fold {fold_idx+1}/{n_splits} -> "
              f"Train: {len(train_subset)} | Val: {len(val_subset)}")

        folds.append((train_loader, val_loader))

    return folds


# =========================================================
# Holdout: treino / validação / teste (70/15/15 por padrão)
# =========================================================

def get_holdout_loaders(
    root_dir,
    batch_size=8,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Divide o dataset em 3 partes:

        - Treino
        - Validação
        - Teste
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \

    transform = default_transform()
    full_dataset = NEUDataset(root_dir=root_dir, transform=transform)

    n_total = len(full_dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"[get_holdout_loaders] Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader
