"""Model architectures for EuroSAT classification."""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights, ResNet50_Weights, ViT_B_16_Weights


class MLP(nn.Module):
    """Simple MLP baseline model."""

    def __init__(self, input_size: int, hidden_layers: list, num_classes: int, dropout: float):
        super().__init__()
        layers = []
        in_features = input_size * input_size * 3

        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


class VGG16Classifier(nn.Module):
    """VGG16 with custom classification head."""

    def __init__(self, num_classes: int, freeze_backbone: bool, dropout: float):
        super().__init__()
        self.backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        num_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class ResNet50Classifier(nn.Module):
    """ResNet50 with custom classification head."""

    def __init__(self, num_classes: int, freeze_backbone: bool):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        self.backbone.fc.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class ViTClassifier(nn.Module):
    """Vision Transformer with custom classification head."""

    def __init__(self, num_classes: int, freeze_backbone: bool):
        super().__init__()
        self.backbone = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(num_features, num_classes)
        self.backbone.heads.head.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def create_model(model_name: str, model_params: dict) -> nn.Module:
    """Factory function to create models."""
    if model_name == "mlp":
        return MLP(
            input_size=model_params["input_size"],
            hidden_layers=model_params["hidden_layers"],
            num_classes=model_params["num_classes"],
            dropout=model_params["dropout"]
        )
    elif model_name == "vgg16":
        return VGG16Classifier(
            num_classes=model_params["num_classes"],
            freeze_backbone=model_params["freeze_backbone"],
            dropout=model_params["dropout"]
        )
    elif model_name == "resnet50":
        return ResNet50Classifier(
            num_classes=model_params["num_classes"],
            freeze_backbone=model_params["freeze_backbone"]
        )
    elif model_name == "vit_b_16":
        return ViTClassifier(
            num_classes=model_params["num_classes"],
            freeze_backbone=model_params["freeze_backbone"]
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
