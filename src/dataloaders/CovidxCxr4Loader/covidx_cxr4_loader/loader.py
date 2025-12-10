import os
import kagglehub
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

class CovidxCxr4Dataset(Dataset):
    def __init__(self, split: str = "train", transform=None):
        """
        Se dataset_list for fornecido, ele usa essa lista (lista de tuples (path,label)).
        Caso contrário, faz o download e carrega o split (train/val/test) do disco.
        """
        assert split in ["train", "test", "val"]

        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomResizedCrop(448, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.data_path = self.download_kaggle_dataset()
        self.dataset = self.load_dataset(split)

    def download_kaggle_dataset(self):
        return kagglehub.dataset_download("andyczhao/covidx-cxr2")

    def load_dataset(self, split):
        images_path = os.path.join(self.data_path, split)
        labels_path = os.path.join(self.data_path, f"{split}.txt")

        dataset = []
        with open(labels_path) as f:
            for line in f:
                _, image_file, label, _ = line.strip().split()
                if label not in ["positive", "negative"]:
                    continue
                label = 1 if label == "positive" else 0
                image_path = os.path.join(images_path, image_file)
                if os.path.exists(image_path):
                    dataset.append((image_path, label))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


def build_loaders(batch_size=16, num_workers=4, train_transforms = None, test_transforms = None):
    # Carrega train completo (para treino)
    train_transform = train_transforms or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomResizedCrop(448, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    train_ds = CovidxCxr4Dataset("train", transform=train_transform)

    # Carrega o "test" publicado e o divide em val/test estratificado
    test_transform = test_transforms or transforms.Compose([
        transforms.Resize((512,512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    test_ds = CovidxCxr4Dataset("test", transform=test_transform)
    val_ds = CovidxCxr4Dataset("test", transform=test_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Weighted sampler para o treino (mantém balanceamento por batch)
    train_labels = torch.tensor([y for _, y in train_ds.dataset])
    class_count = torch.bincount(train_labels.long())

    # weights = 1.0 / class_count.float()
    # sample_weights = weights[train_labels.long()]

    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # =========================
    # pos_weight (para BCEWithLogitsLoss)
    # =========================
    neg, pos = class_count[0].item(), class_count[1].item()
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32)


    return train_loader, val_loader, test_loader, pos_weight
