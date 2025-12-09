import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from models.ae_mlp_classifier import AEMLPClassifier


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_features_split(results_dir: str, split: str):
    path = os.path.join(results_dir, f"features_{split}.pt")
    data = torch.load(path)
    features = data["features"].float()  # [N, 256]
    labels = data["labels"].long()      # [N]
    classes = data["classes"]
    return features, labels, classes


def build_dataloaders(results_dir: str, batch_size: int = 128):
    X_train, y_train, classes = load_features_split(results_dir, "train")
    X_val, y_val, _ = load_features_split(results_dir, "val")

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader}, classes, X_train.shape[1]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for feats, labels in loader:
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * feats.size(0)
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
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for feats, labels in loader:
        feats = feats.to(device)
        labels = labels.to(device)

        outputs = model(feats)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * feats.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")

    return {"loss": epoch_loss, "acc": epoch_acc, "f1": epoch_f1}


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "autoencoder")
    clf_dir = os.path.join(results_dir, "classifier")
    os.makedirs(clf_dir, exist_ok=True)

    device = get_device()
    print("Device:", device)

    loaders, classes, in_dim = build_dataloaders(results_dir, batch_size=128)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    num_classes = len(classes)
    print("Num classes:", num_classes)
    print("Input feature dim:", in_dim)

    model = AEMLPClassifier(in_dim=in_dim, num_classes=num_classes, hidden_dim=256, dropout_p=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 20
    best_val_f1 = 0.0
    ckpt_path = os.path.join(clf_dir, "best_classifier.pt")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

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
                    "classes": classes,
                },
                ckpt_path,
            )
            print(f"  Novo melhor classificador salvo em {ckpt_path} (val_f1={best_val_f1:.4f})")


if __name__ == "__main__":
    main()
