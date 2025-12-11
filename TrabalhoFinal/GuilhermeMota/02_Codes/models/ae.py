import torch

from torch import nn
import torch.nn.functional as F



class AutoEncoder(nn.Module):
    def __init__(self, in_features, num_l1, num_l2, num_l3, ae_type='normal'):
        super().__init__()

        self.ae_type = ae_type

        self.enc1 = nn.Linear(in_features, num_l1)
        self.enc2 = nn.Linear(num_l1, num_l2)
        self.latent = nn.Linear(num_l2, num_l3)
        
        self.dec1 = nn.Linear(num_l3, num_l2)
        self.dec2 = nn.Linear(num_l2, num_l1)
        self.out = nn.Linear(num_l1, in_features)

    
    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))

        if self.ae_type == 'normal':
            z = self.latent(x)
        elif self.ae_type == 'sparse':
            z = torch.sigmoid(self.latent(x))

        return z
    

    def decode(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.out(x)

        return x


    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        
        return x_recon, z