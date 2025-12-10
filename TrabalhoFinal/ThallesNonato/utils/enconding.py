import numpy as np
from torch.utils.data import DataLoader


def encode_ids(train_user_data, train_movie_data,
               test_user_data, test_movie_data):
    """
    Recebe arrays brutos de user_id e movie_id (treino e teste)
    e converte para índices consecutivos (0..N-1).

    Usuários/filmes que aparecem apenas no teste recebem índice 0.
    """


    unique_users = np.unique(train_user_data)
    user2idx = {u: i for i, u in enumerate(unique_users)}

    encoded_train_users = np.array([user2idx[u] for u in train_user_data])
    encoded_test_users = np.array([user2idx.get(u, 0) for u in test_user_data])


    unique_movies = np.unique(train_movie_data)
    movie2idx = {m: i for i, m in enumerate(unique_movies)}

    encoded_train_movies = np.array([movie2idx[m] for m in train_movie_data])
    encoded_test_movies = np.array([movie2idx.get(m, 0) for m in test_movie_data])


    num_users = len(user2idx)
    num_movies = len(movie2idx)

    return (
        encoded_train_users,
        encoded_train_movies,
        encoded_test_users,
        encoded_test_movies,
        num_users,
        num_movies,
        user2idx,
        movie2idx
    )
