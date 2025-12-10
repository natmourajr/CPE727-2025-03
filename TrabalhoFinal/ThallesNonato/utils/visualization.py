import matplotlib.pyplot as plt
import pickle


def plot_movies_by_year(movie_titles):
    data = movie_titles["Ano"].value_counts().sort_index()

    plt.figure(figsize=(12,6))
    plt.plot(data.index, data.values, marker='o')

    plt.title("Quantidade de Filmes por Ano de Lançamento")
    plt.xlabel("Ano")
    plt.ylabel("Número de Filmes")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()


def plot_rating_distribution(df):
    data = df["Rating"].value_counts().sort_index()

    plt.figure(figsize=(12,6))
    bars = plt.bar(data.index, data.values)

    total = df.shape[0]
    for bar in bars:
        h = bar.get_height()
        pct = h / total * 100
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h,
            f"{pct:.1f}%",
            ha="center",
            va="bottom"
        )

    plt.title("Distribuição das Avaliações dos Usuários")
    plt.xlabel("Nota")
    plt.ylabel("Quantidade")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.show()

def plot_ratings_by_day(df):
    data = df["Date"].value_counts()
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, marker="o", color="#db0000")

    plt.title(f"Total de {df.shape[0]} avaliações agrupadas por dia")
    plt.xlabel("Data")
    plt.ylabel("Quantidade de avaliações")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_ratings_per_movie(df, movie_clip=9999):
    data_movie = df.groupby("Movie")["Rating"].count().clip(upper=movie_clip)

    plt.figure(figsize=(12, 6))
    plt.hist(
        data_movie.values,
        bins=range(0, movie_clip + 1, 100),
        color="#db0000",
        edgecolor="black",
        rwidth=0.8,
    )
    
    plt.title("Distribuição de avaliações por filme")
    plt.xlabel("Avaliações por filme")
    plt.ylabel("Quantidade de filmes")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

def plot_ratings_per_user(df, user_clip=199):
    data_user = df.groupby("User")["Rating"].count().clip(upper=user_clip)

    plt.figure(figsize=(12, 6))
    plt.hist(
        data_user.values,
        bins=range(0, user_clip + 2, 2),
        color="#db0000",
        edgecolor="black",
        rwidth=0.8,
    )

    plt.title("Distribuição de avaliações por usuário")
    plt.xlabel("Avaliações por usuário")
    plt.ylabel("Quantidade de usuários")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()


def filter_sparse_data(df, min_movie_ratings=10000, min_user_ratings=200):
    filter_movies = (
        df["Movie"].value_counts() > min_movie_ratings
    )
    filter_movies = filter_movies[filter_movies].index.tolist()

    filter_users = (
        df["User"].value_counts() > min_user_ratings
    )
    filter_users = filter_users[filter_users].index.tolist()

    df_filtered = df[
        (df["Movie"].isin(filter_movies)) &
        (df["User"].isin(filter_users))
    ]

    print(f"Shape antes do filtro: {df.shape}")
    print(f"Shape depois do filtro: {df_filtered.shape}")

    return df_filtered

def load_preprocessed():
    with open("datasets/processed/preprocessed.pkl", "rb") as f:
        return pickle.load(f)