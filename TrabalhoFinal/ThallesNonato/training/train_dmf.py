import torch
import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def train(model, train_loader, optimizer, device, epochs=5, val_loader=None):
    criterion = torch.nn.MSELoss()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ---- TREINO ----
        model.train()
        running_loss = 0.0
        for user, movie, rating in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
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

        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {epoch_loss:.4f}")

        # ---- VALIDAÇÃO ----
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for user, movie, rating in val_loader:
                    user = user.to(device)
                    movie = movie.to(device)
                    rating = rating.to(device)

                    pred = model(user, movie)
                    loss = criterion(pred, rating)
                    val_running_loss += loss.item() * len(user)

            val_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} | Val MSE: {val_loss:.4f}")

    return train_losses, val_losses


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

def precision_at_k(recommended_items, true_items, K):
    recommended_k = recommended_items[:K]
    return len(np.intersect1d(recommended_k, true_items)) / K

def ndcg_at_k(recommended_items, true_items, K):
    recommended_k = recommended_items[:K]
    dcg = 0
    for i, item in enumerate(recommended_k):
        if item in true_items:
            dcg += 1 / np.log2(i + 2)  # +2 porque log2(1) = 0
    # Ideal DCG
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(K, len(true_items))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

def evaluate_ranking_dmf(model, train_users, train_movies, test_users, test_movies, num_users, num_movies, device, K=10):
    """
    Avaliação realista de DMF:
    - candidatos = filmes não vistos no treino
    - itens verdadeiros = filmes do teste
    """
    model.eval()
    df_train = np.column_stack((train_users, train_movies))
    df_test  = np.column_stack((test_users, test_movies))

    precisions, ndcgs = [], []

    with torch.no_grad():
        for user_idx in np.unique(test_users):
            # Filmes do treino
            train_items = df_train[df_train[:,0]==user_idx][:,1]

            # Filmes candidatos = todos menos os do treino
            candidate_items = np.setdiff1d(np.arange(num_movies), train_items)

            # Filmes verdadeiros do teste
            true_items = df_test[df_test[:,0]==user_idx][:,1]

            if len(true_items) == 0 or len(candidate_items) == 0:
                continue  # pular usuários sem teste ou sem candidatos

            # Predições do modelo
            user_id = torch.tensor([user_idx]*len(candidate_items), dtype=torch.long).to(device)
            movie_id = torch.tensor(candidate_items, dtype=torch.long).to(device)
            pred = model(user_id, movie_id).cpu().numpy()

            # Top-K
            recommended_items = candidate_items[np.argsort(pred)[::-1]][:K]

            precisions.append(precision_at_k(recommended_items, true_items, K))
            ndcgs.append(ndcg_at_k(recommended_items, true_items, K))

    precision_mean = np.mean(precisions)
    ndcg_mean = np.mean(ndcgs)

    print(f"➡️ Precision@{K}: {precision_mean:.4f}, NDCG@{K}: {ndcg_mean:.4f}")
    return precision_mean, ndcg_mean
