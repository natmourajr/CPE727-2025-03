from torchvision import models
from torch import nn

class DenseNet121(nn.Module):
    def __init__(self, num_classes: int = 1, device='cpu', pretrained: bool = True):
        super(DenseNet121, self).__init__()

        # Carregar pesos do ImageNet
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.densenet121(weights=weights)


        # Alterar o classificador para o número de classes desejado
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

        self.model.to(device)

        # Freeze backbone
        for p in self.model.features.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True


    def forward(self, x):
        return self.model(x)