import torch
import tqdm

def train(model, train_loader, optimizer, device, epochs=5):
    criterion = torch.nn.MSELoss()

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

        print(f"Epoch {epoch+1}/{epochs} | MSE: {running_loss / len(train_loader.dataset):.4f}")

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

