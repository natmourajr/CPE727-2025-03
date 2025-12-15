import torch
import numpy as np

from models.ncf import NCF   # ajuste o caminho para onde seu NCF está


# ==========================
# 1) CARREGAR O MODELO SALVO
# ==========================

num_users  = n_users_usado_no_treino
num_movies = n_movies_usado_no_treino

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NCF(
    n_users=num_users,
    n_items=num_movies,
    n_factors=32,
    hidden_dims=[64,32,16,8]
).to(device)

state = torch.load("ncf.pth", map_location=device)
model.load_state_dict(state)
model.eval()

print("Modelo carregado!")


# ==============================================
# 2) CARREGAR ARRAYS DO TREINO / TEST (NUMPY)
# ==============================================

train_users  = np.load("train_users.npy")
train_movies = np.load("train_movies.npy")

test_users   = np.load("test_users.npy")
test_movies  = np.load("test_movies.npy")

print("Dados carregados!")


# ===============================
# 3) AVALIAR RANKING
# ===============================

precision, ndcg = evaluate_ranking(
    model,
    train_users,
    train_movies,
    test_users,
    test_movies,
    num_users,
    num_movies,
    device,
    K=10
)

print(f"\nPrecision@10 = {precision:.4f}")
print(f"NDCG@10      = {ndcg:.4f}")
