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


class SimpleCNN(nn.Module):
    """
    CNN Simples treinada do zero (sem transfer learning) para servir como baseline
    
    Arquitetura leve e eficiente para comparação com modelos pré-treinados
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        """
        Args:
            num_classes: Número de classes (2 para binário)
            dropout: Taxa de dropout para regularização
        """
        super(SimpleCNN, self).__init__()
        
        self.model_name = 'simplecnn'
        self.num_classes = num_classes
        
        # Bloco Convolucional 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )
        
        # Bloco Convolucional 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )
        
        # Bloco Convolucional 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )
        
        # Bloco Convolucional 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Inicialização dos pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa os pesos da rede"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def freeze_backbone(self):
        """Método vazio para compatibilidade com TBClassifier"""
        pass
    
    def unfreeze_backbone(self):
        """Método vazio para compatibilidade com TBClassifier"""
        pass


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
        model_name: Nome do modelo ('resnet50', 'densenet121', 'efficientnet_b0', 'simplecnn')
        pretrained: Se deve usar pesos pré-treinados (ignorado para simplecnn)
        num_classes: Número de classes
        dropout: Taxa de dropout
        
    Returns:
        Modelo PyTorch
    """
    if model_name == 'simplecnn':
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    else:
        return TBClassifier(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout
        )
