from torch.utils.data import DataLoader
from datasets.NetflixDataset import NetflixDataset
from datasets.UserDataset import UserDataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch

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

def create_autorec_dataloader(train_matrix, batch_size=64):
    R_filled = np.nan_to_num(train_matrix, nan=0.0)
    R_tensor = torch.tensor(R_filled, dtype=torch.float32)

    dataset = UserDataset(R_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

