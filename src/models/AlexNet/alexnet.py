from torchvision import models, transforms
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_channels: int = 3, num_classes: int = 10, device: str = 'cpu', pretrained: bool = True):
        super().__init__()
        self.num_channels = num_channels

        # Carrega modelo pré-treinado
        weights = None
        if pretrained:
            weights = models.AlexNet_Weights.IMAGENET1K_V1
        self.model = models.alexnet(weights=weights)

        # Ajusta camada de entrada se MNIST (1 canal)
        self.model.features[0] = nn.Conv2d(
            3, 64, kernel_size=11, stride=4, padding=2
        )

        # Substitui classificador final
        self.model.features[0] = nn.Conv2d(self.num_channels, 64, kernel_size=11, stride=4, padding=2)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

        self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def default_dataloader_transforms(self):
        transforms_base = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        preprocess = transforms.Compose(transforms_base)
        return preprocess
