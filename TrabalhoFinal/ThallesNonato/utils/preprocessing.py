import pandas as pd
import numpy as np
from collections import deque


def load_movie_titles(path="dataset/movie_titles.csv"):
    movie_titles = pd.read_csv(
        path,
        encoding='ISO-8859-1',
        header=None,
        index_col=0,
        names=['Id', 'Ano', 'Parte1', 'Parte2', 'Parte3', 'Parte4', 'Parte5'],
        engine='python'
    )

    movie_titles["Nome"] = movie_titles[['Parte1','Parte2','Parte3','Parte4','Parte5']] \
        .fillna('') \
        .agg(','.join, axis=1) \
        .str.replace(r",+$", "", regex=True)

    movie_titles = movie_titles.drop(columns=['Parte1','Parte2','Parte3','Parte4','Parte5'])

    return movie_titles


def load_raw_ratings(path="dataset/combined_data_1.txt"):
    df_raw = pd.read_csv(
        path, 
        header=None,
        names=["User", "Rating", "Date"],
        usecols=[0, 1, 2]
    )
    return df_raw


def parse_netflix_ratings(df_raw):
    """
    Converte o formato da Netflix:
    MovieID:
    UserID, Rating, Date
    UserID, Rating, Date
    MovieID:
    ...

    → para um dataframe com colunas:
    User, Rating, Date, Movie
    """

    tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
    movie_indices = [[idx, int(mid[:-1])] for idx, mid in tmp_movies.values]

    shifted = deque(movie_indices)
    shifted.rotate(-1)

    user_blocks = []

    for (i1, movie_id), (i2, next_movie) in zip(movie_indices, shifted):

        if i1 < i2:
            block = df_raw.loc[i1+1 : i2-1].copy()
        else:
            block = df_raw.loc[i1+1:].copy()

        block["Movie"] = movie_id
        user_blocks.append(block)

    df = pd.concat(user_blocks)
    df = df.dropna(subset=["Rating"])

    return df

def filter_sparse(df, min_movie_ratings=10000, min_user_ratings=200):
    """
    Filtra filmes e usuários com poucas avaliações.
    """
    valid_movies = df['Movie'].value_counts()
    valid_movies = valid_movies[valid_movies > min_movie_ratings].index.tolist()

    valid_users = df['User'].value_counts()
    valid_users = valid_users[valid_users > min_user_ratings].index.tolist()

    df_filtered = df[(df['Movie'].isin(valid_movies)) &
                     (df['User'].isin(valid_users))]

    return df_filtered


def create_train_test(df_filtered):
    """
    Gera X_train e X_test no formato pivotado.
    """
    df_pivot = df_filtered.pivot_table(index='User', columns='Movie', values='Rating')
    X_train, X_test = train_test_split(df_pivot, test_size=0.2, random_state=42)

    return X_train, X_test


def prepare_data(df):

    df_p = df.pivot_table(index="User", columns="Movie", values="Rating")

    X_train, X_test = train_test_split(df_p, test_size=0.2, random_state=42)

    # desfaz pivot
    train_df = X_train.stack().reset_index()
    train_df.columns = ["User", "Movie", "Rating"]

    test_df = X_test.stack().reset_index()
    test_df.columns = ["User", "Movie", "Rating"]

    # encode users
    unique_users = np.unique(train_df["User"])
    user2idx = {u: i for i, u in enumerate(unique_users)}

    # encode movies
    unique_movies = np.unique(train_df["Movie"])
    movie2idx = {m: i for i, m in enumerate(unique_movies)}

    # train
    train_users = np.array([user2idx[u] for u in train_df["User"]])
    train_movies = np.array([movie2idx[m] for m in train_df["Movie"]])
    train_ratings = train_df["Rating"].astype(np.float32).values

    # test
    test_users = np.array([user2idx.get(u, 0) for u in test_df["User"]])
    test_movies = np.array([movie2idx.get(m, 0) for m in test_df["Movie"]])
    test_ratings = test_df["Rating"].astype(np.float32).values

    return (
        train_users, train_movies, train_ratings,
        test_users, test_movies, test_ratings,
        len(user2idx), len(movie2idx)
    )

