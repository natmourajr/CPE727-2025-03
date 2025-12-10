import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def train_autorec(model, dataloader, optimizer, device, epochs=10):
    model.train()
    criterion = nn.MSELoss(reduction="mean")
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0

        for batch in dataloader:  
            batch = batch.to(device)

            optimizer.zero_grad()

            output = model(batch)

            mask = (batch != 0).float()
            loss = criterion(output * mask, batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        train_losses.append(epoch_loss)

        print(f"[AutoRec] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    return train_losses

def evaluate_autorec(model, test_matrix, device=None):
    model.eval()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_matrix = torch.tensor(test_matrix, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(test_matrix)

    output = output.cpu().numpy()
    true_values = test_matrix.cpu().numpy()

    mask = ~np.isnan(true_values)

    mse = np.mean((output[mask] - true_values[mask]) ** 2)
    rmse = np.sqrt(mse)

    print(f"\n➡️ AutoRec Test RMSE: {rmse:.4f}")

    return rmse

