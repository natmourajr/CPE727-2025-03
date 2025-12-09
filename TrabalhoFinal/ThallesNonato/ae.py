from torch import nn
import torch.nn.functional as F

class AutoRec(nn.Module):
    def __init__(self, num_items, hidden_dim=500):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_items, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_items)
        self.activation = nn.ReLU()  

    def forward(self, x):
        z = self.activation(self.encoder(x))
        out = self.decoder(z) 
        return out