from torchvision import models
from torch import nn

class ViTBinaryClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        device="cpu",
        pretrained: bool = True,
    ):
        super().__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.vit_b_16(weights=weights)

        embed_dim = self.model.heads.head.in_features
        self.model.heads = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

        self.to(device)

        # Freeze backbone (vit sem head)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.heads.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)
