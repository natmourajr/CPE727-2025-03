import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# 1. Cria matriz User × Movie (tabela pivô)
# =========================================================
def create_user_movie_matrix(df):
    """
    Recebe um DataFrame com colunas ['User', 'Movie', 'Rating'].
    Retorna a matriz pivô User × Movie com valores de rating.
    """
    return df.pivot_table(index="User", columns="Movie", values="Rating")


# =========================================================
# 2. Divide a matriz em treino e teste (por usuários)
# =========================================================
def split_matrix(df_pivot, test_size=0.2, random_state=42):
    """
    Divide a matriz pivô mantendo todas as colunas (filmes),
    mas separando usuários entre treino e teste.
    """
    X_train, X_test = train_test_split(
        df_pivot,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test


# =========================================================
# 3. Converte matriz pivô em linhas (User, Movie, Rating)
# =========================================================
def flatten_matrix(df_matrix):
    """
    Converte uma matriz User × Movie em um DataFrame com colunas
    ['User', 'Movie', 'Rating'], removendo NaN.
    """
    df_flat = df_matrix.stack().reset_index()
    df_flat.columns = ["User", "Movie", "Rating"]
    return df_flat


# =========================================================
# 4. Converte DataFrame para tensores/arrays usados no modelo
# =========================================================
def prepare_tensors(df_flat):
    """
    Transforma DataFrame (User, Movie, Rating) em arrays numpy:
      - user_ids   (int64)
      - movie_ids  (int64)
      - ratings    (float32)
    """
    user_data  = df_flat["User"].astype(np.int64).values
    movie_data = df_flat["Movie"].astype(np.int64).values
    ratings    = df_flat["Rating"].astype(np.float32).values

    return user_data, movie_data, ratings
