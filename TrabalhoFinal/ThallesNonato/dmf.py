from torch import nn, cat
import torch
class DMF(nn.Module):
    def __init__(self, num_users, num_movies, user_emb_size=20, movie_emb_size=10):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_emb_size)
        self.movie_embedding = nn.Embedding(num_movies, movie_emb_size)
        
        self.fc1 = nn.Linear(user_emb_size + movie_emb_size, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, user_id, movie_id):
        user_vec = self.user_embedding(user_id)
        movie_vec = self.movie_embedding(movie_id)
        x = torch.cat([user_vec, movie_vec], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.squeeze()

