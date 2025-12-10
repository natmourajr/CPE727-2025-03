import torch
import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error


def train(model, train_loader, optimizer, device, epochs=5):
    criterion = torch.nn.MSELoss()
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for user, movie, rating in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            user = user.to(device)
            movie = movie.to(device)
            rating = rating.to(device)

            optimizer.zero_grad()
            pred = model(user, movie)
            loss = criterion(pred, rating)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(user)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs} | MSE: {epoch_loss:.4f}")

    return train_losses  


def evaluate_dmf(model, test_loader, device):
    model.eval()
    y_pred, y_true = [], []

    with torch.no_grad():
        for user, movie, rating in test_loader:
            user = user.to(device)
            movie = movie.to(device)
            rating = rating.to(device)

            pred = model(user, movie)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(rating.cpu().numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n➡️ Test RMSE: {rmse:.4f}")

    return rmse
