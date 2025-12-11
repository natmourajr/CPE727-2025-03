import sys, os, copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

from src.models.SAE_NEU.model import SAE_NEU
from src.dataloaders.NEU_loader.loader import NEUDataset, default_transform


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for imgs, _ in tqdm(dataloader, desc="Treinando AE", leave=False):
        imgs = imgs.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def plot_loss(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Treino", marker="o")
    plt.plot(epochs, history["val_loss"], label="Validação", marker="o")
    plt.xlabel("Época")
    plt.ylabel("Loss (MSE)")
    plt.title("Autoencoder NEU-DET - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Usando dispositivo: {device}")
    torch.backends.cudnn.benchmark = True

    root_dir = os.path.join("Data", "NEU-DET")
    save_dir = os.path.join("src", "Results", "SAE_NEU")
    os.makedirs(save_dir, exist_ok=True)

    transform = default_transform()
    dataset = NEUDataset(root_dir=root_dir, transform=transform)
    n_total = len(dataset)
    labels = [label for _, label in dataset.samples]
    print(f"[Dataset] Total de amostras: {n_total}")

    train_ratio = 0.85
    seed = 42
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(np.arange(n_total), labels))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    print(f"[Split AE] Train: {len(train_idx)} | Val: {len(val_idx)}")

    model = SAE_NEU(latent_dim=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    max_epochs = 100
    patience = 10

    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[Época {epoch+1}/{max_epochs}] "
              f"Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" Early stopping na época {epoch+1}")
                break

    best_model_path = os.path.join(save_dir, "autoencoder_best.pth")
    torch.save(best_state, best_model_path)
    print(f"Autoencoder salvo em: {best_model_path}")

    model.load_state_dict(best_state)
    encoder_state = model.encoder.state_dict()
    encoder_path = os.path.join(save_dir, "encoder_best.pth")
    torch.save(encoder_state, encoder_path)
    print(f" Encoder salvo em: {encoder_path}")

    plot_path = os.path.join(save_dir, "sae_loss.png")
    plot_loss(history, plot_path)
    print(f"Gráfico de loss salvo em: {plot_path}")

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "sae_loss_history.csv"), index_label="epoch")
    print(f"Histórico de loss salvo em: {save_dir}")


if __name__ == "__main__":
    main()
