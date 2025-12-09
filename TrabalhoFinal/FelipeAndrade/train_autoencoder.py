import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from datasets.mbgv2_crops_dataset import Mbgv2CropsDataset
from models.conv_autoencoder import ConvAutoencoder


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(batch_size: int = 64) -> Dict[str, DataLoader]:
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),   # [0,1]
    ])

    train_ds = Mbgv2CropsDataset(split="train", transform=transform)
    val_ds = Mbgv2CropsDataset(split="val", transform=transform)

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
) -> float:
    model.train()
    running_loss = 0.0

    for imgs, _ in loader:  # labels não são usados no AE
        imgs = imgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        recon = model(imgs)
        loss = criterion(recon, imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        recon = model(imgs)
        loss = criterion(recon, imgs)
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "autoencoder")
    os.makedirs(results_dir, exist_ok=True)

    device = get_device()
    print("Device:", device)

    loaders = build_dataloaders(batch_size=64)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    model = ConvAutoencoder(latent_dim=256).to(device)

    # MSE para reconstrução
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    num_epochs = 15
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(results_dir, "best_ae.pt")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = evaluate(
            model, val_loader, criterion, device
        )

        print(f"  Train recon loss: {train_loss:.6f}")
        print(f"  Val   recon loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )
            print(
                f"  Novo melhor AE salvo em {best_ckpt_path} "
                f"(val_loss={best_val_loss:.6f})"
            )


if __name__ == "__main__":
    main()
