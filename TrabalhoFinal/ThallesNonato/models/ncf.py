import torch
from torch import nn

class NCF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=32, hidden_dims=[64,32,16,8]):
        super().__init__()
        self.user_emb_gmf = nn.Embedding(n_users, n_factors)
        self.item_emb_gmf = nn.Embedding(n_items, n_factors)
        self.user_emb_mlp = nn.Embedding(n_users, n_factors)
        self.item_emb_mlp = nn.Embedding(n_items, n_factors)

        mlp_layers = []
        input_size = n_factors * 2
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(input_size, h))
            mlp_layers.append(nn.ReLU())
            input_size = h
        
        self.mlp = nn.Sequential(*mlp_layers)

        self.output = nn.Linear(n_factors + hidden_dims[-1], 1)

    def forward(self, user, item):
        gmf = self.user_emb_gmf(user) * self.item_emb_gmf(item)
        mlp = torch.cat([self.user_emb_mlp(user), self.item_emb_mlp(item)], dim=1)
        mlp = self.mlp(mlp)
        x = torch.cat([gmf, mlp], dim=1)
        return self.output(x).squeeze()

