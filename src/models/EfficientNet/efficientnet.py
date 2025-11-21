from torchvision import models, transforms
from torch import nn

class EfficientNetB0(nn.Module):
    def __init__(self, num_channels: int = 3, num_classes: int = 10, device = 'cpu', pretrained: bool = True):
        super(EfficientNetB0, self).__init__()

        weights = None
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)

        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)

        self.model.features[0][0] = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.model.to(device)

    def forward(self, x):
        return self.model(x)
    
    def default_dataloader_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),  # EfficientNet espera 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # normalização simples
        ])