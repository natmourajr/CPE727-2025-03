import torch
from sklearn.metrics import mean_squared_error
import numpy as np
import tqdm

import torch
from torch import nn, optim
import tqdm

def train_ncf(model, train_loader, valid_loader=None, epochs=100, lr=0.001, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_losses = []
    val_losses = []

    # --- Early stopping ---
    best_val = float("inf")
    wait = 0
    patience = 8
    best_state = None

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

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

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

            # --- Early stopping check ---
            if avg_val_loss < best_val - 1e-4:
                best_val = avg_val_loss
                wait = 0
                best_state = model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    model.load_state_dict(best_state)
                    break

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

def evaluate_ranking(model, train_users, train_movies, test_users, test_movies, num_users, num_movies, device, K=10):
    """
    Avaliação de ranking (Precision@K e NDCG@K) para modelos de recomendação
    que recebem (user_id, item_id) como entrada, como DMF ou NCF.

    Args:
        model: modelo treinado
        train_users, train_movies: arrays de treino
        test_users, test_movies: arrays de teste (reduzidos)
        num_users, num_movies: total de usuários e itens
        device: cuda ou cpu
        K: top-K para ranking
    """
    model.eval()
    df_train = np.column_stack((train_users, train_movies))
    df_test  = np.column_stack((test_users, test_movies))

    precisions, ndcgs = [], []

    def precision_at_k(recommended, true_items, K):
        recommended_k = recommended[:K]
        return len(np.intersect1d(recommended_k, true_items)) / K

    def ndcg_at_k(recommended, true_items, K):
        recommended_k = recommended[:K]
        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended_k) if item in true_items)
        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(K, len(true_items))))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0

    with torch.no_grad():
        for user_idx in np.unique(test_users):
            train_items = df_train[df_train[:,0]==user_idx][:,1]
            candidate_items = np.setdiff1d(np.arange(num_movies), train_items)
            true_items = df_test[df_test[:,0]==user_idx][:,1]

            if len(true_items) == 0 or len(candidate_items) == 0:
                continue

            user_tensor = torch.tensor([user_idx]*len(candidate_items), dtype=torch.long).to(device)
            movie_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)

            preds = model(user_tensor, movie_tensor).cpu().numpy()
            recommended_items = candidate_items[np.argsort(preds)[::-1]][:K]

            precisions.append(precision_at_k(recommended_items, true_items, K))
            ndcgs.append(ndcg_at_k(recommended_items, true_items, K))

    precision_mean = np.mean(precisions)
    ndcg_mean = np.mean(ndcgs)
    return precision_mean, ndcg_mean
