import os
from typing import Tuple, Dict, List

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from datasets.mbgv2_crops_dataset import Mbgv2CropsDataset
from models.conv_autoencoder import ConvAutoencoder


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(split: str, batch_size: int = 64) -> Tuple[DataLoader, List[str]]:
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    dataset = Mbgv2CropsDataset(split=split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader, dataset.classes


@torch.no_grad()
def extract_features_for_split(
    model: ConvAutoencoder,
    split: str,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    loader, classes = build_loader(split, batch_size=batch_size)

    all_feats = []
    all_labels = []

    print(f"\nExtraindo features para split='{split}'...")

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)

        z = model.encode(imgs)  # [B, latent_dim]
        all_feats.append(z.cpu())
        all_labels.append(labels.clone())

    feats_tensor = torch.cat(all_feats, dim=0)   # [N, latent_dim]
    labels_tensor = torch.cat(all_labels, dim=0) # [N]

    print(f"Split '{split}': features shape = {feats_tensor.shape}, labels shape = {labels_tensor.shape}")

    return {
        "features": feats_tensor,
        "labels": labels_tensor,
        "classes": classes,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "autoencoder")
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(results_dir, "best_ae.pt")
    print("Carregando autoencoder de:", ckpt_path)

    device = get_device()
    print("Device:", device)

    # Mesmo latent_dim usado no treino do AE
    model = ConvAutoencoder(latent_dim=256).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint do AE treinado até epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")

    for split in ["train", "val", "test"]:
        data = extract_features_for_split(model, split, device=device, batch_size=64)
        out_path = os.path.join(results_dir, f"features_{split}.pt")
        torch.save(data, out_path)
        print(f"Features de '{split}' salvas em {out_path}")


if __name__ == "__main__":
    main()
