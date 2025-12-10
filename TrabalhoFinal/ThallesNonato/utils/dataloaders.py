from torch.utils.data import DataLoader
from dataset.netflixDataset import NetflixDataset

def create_dataloaders(
    train_users, train_movies, train_ratings,
    test_users, test_movies, test_ratings,
    batch_size=128
):
    """
    Cria datasets e dataloaders prontos para treino e teste.
    """

    train_dataset = NetflixDataset(train_users, train_movies, train_ratings)
    test_dataset  = NetflixDataset(test_users, test_movies, test_ratings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader
