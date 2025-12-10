import torch
from torch.utils.data import Dataset

class UserDataset(Dataset):
    def __init__(self, ratings_matrix):
        self.ratings = torch.tensor(ratings_matrix, dtype=torch.float32)
        
    def __len__(self):
        return self.ratings.shape[0]
    
    def __getitem__(self, idx):
        return self.ratings[idx]