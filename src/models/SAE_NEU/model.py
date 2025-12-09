import torch
import torch.nn as nn


class SAE_NEU(nn.Module):
 

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        in_features = 1 * 200 * 200 

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, in_features),
            nn.Sigmoid()   
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)      # (B, 40000)
        z = self.encoder(x)            # (B, latent_dim)
        recon = self.decoder(z)        # (B, 40000)
        recon = recon.view(x.size(0), 1, 200, 200)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
      
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return z
