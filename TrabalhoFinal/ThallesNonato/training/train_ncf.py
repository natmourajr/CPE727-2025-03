import torch
from sklearn.metrics import mean_squared_error
import numpy as np
import tqdm

import torch
from torch import nn, optim
import tqdm

def train_ncf(model, train_loader, valid_loader=None, epochs=10, lr=0.001, device=None):
    """
    Treina o modelo NCF e opcionalmente avalia na validação a cada época.

    Retorna:
        train_losses: lista com loss média por época no treino
        val_losses: lista com loss média por época na validação (se valid_loader fornecido)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- Treino ---
        model.train()
        total_train_loss = 0
        for user, item, rating in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            user, item, rating = user.to(device), item.to(device), rating.to(device)

            optimizer.zero_grad()
            output = model(user, item)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * user.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # --- Validação ---
        if valid_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for user, item, rating in tqdm.tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} - Val"):
                    user, item, rating = user.to(device), item.to(device), rating.to(device)
                    output = model(user, item)
                    loss = criterion(output, rating)
                    total_val_loss += loss.item() * user.size(0)
            avg_val_loss = total_val_loss / len(valid_loader.dataset)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

    return train_losses, val_losses


def evaluate_ncf(model, data_loader, device=None):
    """
    Avalia o modelo NCF e retorna RMSE.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    preds, truths = [], []

    with torch.no_grad():
        for user, item, rating in data_loader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)

            output = model(user, item)
            preds.extend(output.cpu().numpy())
            truths.extend(rating.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(truths, preds))
    return rmse
