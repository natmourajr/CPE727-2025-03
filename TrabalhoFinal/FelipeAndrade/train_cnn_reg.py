import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score

from datasets.mbgv2_crops_dataset import Mbgv2CropsDataset
from models.cnn_classifier_reg import RegularizedCNN


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(batch_size: int = 64) -> Dict[str, DataLoader]:
    # Data augmentation mais forte no train
    transform_train = T.Compose([
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(degrees=10),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        T.ToTensor(),
    ])

    # Validação só com transform
    transform_eval = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    train_ds = Mbgv2CropsDataset(split="train", transform=transform_train)
    val_ds = Mbgv2CropsDataset(split="val", transform=transform_eval)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return {"train": train_loader, "val": val_loader}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    return {"loss": epoch_loss, "acc": epoch_acc, "f1": epoch_f1}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    return {"loss": epoch_loss, "acc": epoch_acc, "f1": epoch_f1}


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "cnn_reg")
    os.makedirs(results_dir, exist_ok=True)

    device = get_device()
    print("Device:", device)

    loaders = build_dataloaders(batch_size=64)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    num_classes = len(train_loader.dataset.classes)
    model = RegularizedCNN(num_classes=num_classes, dropout_p=0.5).to(device)

    criterion = nn.CrossEntropyLoss()

    # Adam com weight decay (L2)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    num_epochs = 15
    best_val_f1 = 0.0
    best_ckpt_path = os.path.join(results_dir, "best_model.pt")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"  Train - loss: {train_metrics['loss']:.4f}, "
            f"acc: {train_metrics['acc']:.4f}, f1: {train_metrics['f1']:.4f}"
        )
        print(
            f"  Val   - loss: {val_metrics['loss']:.4f}, "
            f"acc: {val_metrics['acc']:.4f}, f1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_val_f1,
                    "epoch": epoch,
                    "classes": train_loader.dataset.classes,
                },
                best_ckpt_path,
            )
            print(
                f"  Novo melhor modelo salvo em {best_ckpt_path} "
                f"(val_f1={best_val_f1:.4f})"
            )


if __name__ == "__main__":
    main()
