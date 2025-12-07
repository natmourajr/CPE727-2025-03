"""
Modelos de Deep Learning para detecção de tuberculose
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class TBClassifier(nn.Module):
    """
    Classificador base para detecção de tuberculose usando transfer learning
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            model_name: Nome do modelo base ('resnet50', 'densenet121', 'efficientnet_b0', 'vgg16')
            pretrained: Se deve usar pesos pré-treinados no ImageNet
            num_classes: Número de classes (2 para binário)
            dropout: Taxa de dropout
        """
        super(TBClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Carregar modelo base
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Modelo {model_name} não suportado")
        
        # Classificador customizado
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        """Congela os parâmetros do backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Descongela os parâmetros do backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class EnsembleModel(nn.Module):
    """
    Ensemble de múltiplos modelos para melhorar a performance
    """
    
    def __init__(
        self,
        model_names: list = ['resnet50', 'densenet121', 'efficientnet_b0'],
        pretrained: bool = True,
        num_classes: int = 2
    ):
        """
        Args:
            model_names: Lista de nomes dos modelos a serem combinados
            pretrained: Se deve usar pesos pré-treinados
            num_classes: Número de classes
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList([
            TBClassifier(
                model_name=name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            for name in model_names
        ])
        
        self.num_models = len(model_names)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - média das predições"""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Média das predições
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        return ensemble_output


def create_model(
    model_name: str = 'resnet50',
    pretrained: bool = True,
    num_classes: int = 2,
    dropout: float = 0.5
) -> nn.Module:
    """
    Factory function para criar modelos
    
    Args:
        model_name: Nome do modelo
        pretrained: Se deve usar pesos pré-treinados
        num_classes: Número de classes
        dropout: Taxa de dropout
        
    Returns:
        Modelo PyTorch
    """
    return TBClassifier(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout
    )
