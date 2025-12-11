import torch
import torch.nn as nn
from src.models.SAE_NEU.model import SAE_NEU


class DNN_SAE_NEU(nn.Module):


    def __init__(self, num_classes: int = 6, latent_dim: int = 32):
        super().__init__()

        sae = SAE_NEU(latent_dim=latent_dim)
        self.encoder = sae.encoder  # reaproveita apenas o encoder

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   
        z = self.encoder(x)        
        logits = self.classifier(z) 
        return logits
