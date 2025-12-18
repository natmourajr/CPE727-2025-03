import torch
import torch.nn as nn
from src.models.SAE_NEU.model import SAE_NEU


class CNN_SAE_NEU(nn.Module):

    def __init__(self, num_classes: int = 6, latent_dim: int = 32):
        super().__init__()

        # Carrega o SAE e usa só o encoder (parte que gera o embedding latente)
        sae = SAE_NEU(latent_dim=latent_dim)
        self.encoder = sae.encoder  # somente o encoder do SAE

        # Extrator de features da CNN
        self.cnn_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 200 -> 100

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 100 -> 50

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 50 -> 25
        )

        # 64 * 25 * 25 = 40000
        self.cnn_fc = nn.Sequential(
            nn.Linear(64 * 25 * 25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # fusão: [features da CNN | embedding do SAE]
        fusion_dim = 128 + latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)

        # Entrada para o SAE: vetoriza a imagem
        x_flat = x.view(b, -1)          # (b, H*W)

        # Saída latente do SAE
        z_sae = self.encoder(x_flat)    # (b, latent_dim)

        # Features da CNN
        feat = self.cnn_features(x)     # (b, 64, 25, 25)
        feat = feat.view(b, -1)         # (b, 64*25*25)
        feat = self.cnn_fc(feat)        # (b, 128)

        # Fusão CNN + SAE
        fused = torch.cat([feat, z_sae], dim=1)  # (b, 128 + latent_dim)

        # Classificador final
        logits = self.classifier(fused)
        return logits
